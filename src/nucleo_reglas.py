import math
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from mlxtend.frequent_patterns import fpgrowth


# Codificación
def matriz_binaria(canastas, items=None, min_conteo=1):
    # Matriz dispersa de transacciones x productos
    todos = np.concatenate(canastas).astype(np.int64)
    if items is None:
        u, c = np.unique(todos, return_counts=True)
        items = u[c >= min_conteo]
    else:
        items = np.unique(np.asarray(items, dtype=np.int64))
    if len(items) == 0:
        return sp.csr_matrix((len(canastas), 0), dtype=bool), items
    filas = np.repeat(np.arange(len(canastas)), [len(x) for x in canastas])
    pos = np.minimum(np.searchsorted(items, todos), len(items) - 1)
    ok = items[pos] == todos
    M = sp.csr_matrix((np.ones(int(ok.sum()), dtype=bool), (filas[ok], pos[ok])),
                      shape=(len(canastas), len(items)))
    return M, items


def a_dataframe(M, items):
    return pd.DataFrame(M.toarray(), columns=[int(i) for i in items])


def conteo_minimo(min_soporte, n):
    # Número mínimo de transacciones equivalente al soporte mínimo
    return max(1, math.ceil(min_soporte * n - 1e-9))


def soporte_efectivo(min_soporte, n):
    # Para que los tres algoritmos usen el mismo umbral entero y no haya diferencias por redondeo
    return (conteo_minimo(min_soporte, n) - 0.5) / n


def normalizar_frecuentes(df_mlxtend):
    return {frozenset(int(x) for x in s): float(v)
            for s, v in zip(df_mlxtend['itemsets'], df_mlxtend['support'])}


# Reglas y métricas
def agregar_metricas(df):
    sa = df['soporte_antecedente'].to_numpy(float)
    sc = df['soporte_consecuente'].to_numpy(float)
    s = df['soporte'].to_numpy(float)
    conf = df['confianza'].to_numpy(float)
    df['lift'] = conf / sc
    df['leverage'] = s - sa * sc
    with np.errstate(divide='ignore', invalid='ignore'):
        df['conviction'] = np.where(conf < 1, (1 - sc) / (1 - conf), np.inf)
    df['jaccard'] = s / (sa + sc - s)
    df['len_antecedente'] = df['antecedente'].map(len).astype(int)
    df['len_consecuente'] = df['consecuente'].map(len).astype(int)
    return df


def generar_reglas(frecuentes, conf_min=0.0, solo_consecuente_unitario=False):
    # Genera A -> C para cada conjunto frecuente (A u C) y cada partición no vacía
    filas = []
    for itemset, s_ac in frecuentes.items():
        k = len(itemset)
        if k < 2:
            continue
        elems = sorted(itemset)
        for r in range(1, k):
            if solo_consecuente_unitario and k - r != 1:
                continue
            for ant in itertools.combinations(elems, r):
                ant_f = frozenset(ant)
                cons_f = itemset - ant_f
                s_a, s_c = frecuentes[ant_f], frecuentes[cons_f]
                conf = s_ac / s_a
                if conf < conf_min:
                    continue
                filas.append((tuple(ant), tuple(sorted(cons_f)), s_a, s_c, s_ac, conf))
    cols = ['antecedente', 'consecuente', 'soporte_antecedente', 'soporte_consecuente', 'soporte', 'confianza']
    return agregar_metricas(pd.DataFrame(filas, columns=cols))


def calcular_mejora(reglas, frecuentes):
    # mejora(A->C) = conf(A->C) - max conf(A'->C) con A' subconjunto propio no vacío de A
    out = []
    for a, c, conf in zip(reglas['antecedente'], reglas['consecuente'], reglas['confianza']):
        if len(a) < 2:
            out.append(np.nan)
            continue
        cf, mejor = frozenset(c), 0.0
        for r in range(1, len(a)):
            for sub in itertools.combinations(a, r):
                sf = frozenset(sub)
                mejor = max(mejor, frecuentes[sf | cf] / frecuentes[sf])
        out.append(conf - mejor)
    return np.array(out, dtype=float)


def frecuentes_fpgrowth(canastas, min_soporte, max_len=None):
    n = len(canastas)
    M, items = matriz_binaria(canastas, min_conteo=conteo_minimo(min_soporte, n))
    if M.shape[1] == 0:
        return {}
    r = fpgrowth(a_dataframe(M, items), min_support=soporte_efectivo(min_soporte, n),
                 use_colnames=True, max_len=max_len)
    return normalizar_frecuentes(r)


def minar(canastas, min_soporte, conf_min=0.0, max_len=3, solo_consecuente_unitario=False):
    fr = frecuentes_fpgrowth(canastas, min_soporte, max_len=max_len)
    reglas = generar_reglas(fr, conf_min, solo_consecuente_unitario)
    reglas['n_transacciones'] = np.rint(reglas['soporte'].to_numpy(float) * len(canastas)).astype(int)
    return reglas, fr


# Base de reglas del recomendador
def _top_por_grupo(grupo, valor, k):
    # Máscara con los k mayores valores de cada grupo
    orden = np.lexsort((-valor, grupo))
    g = grupo[orden]
    inicio = np.r_[0, np.flatnonzero(np.diff(g)) + 1]
    rango = np.arange(len(g)) - np.repeat(inicio, np.diff(np.r_[inicio, len(g)]))
    m = np.zeros(len(grupo), dtype=bool)
    m[orden[rango < k]] = True
    return m


def reglas_pares(canastas, min_conteo_par=5, top_k_conf=None, top_k_lift=None):
    # Todas las reglas 1 -> 1 cuyo par co-ocurre >= min_conteo_par veces
    n = len(canastas)
    M, items = matriz_binaria(canastas, min_conteo=min_conteo_par)
    Mi = M.astype(np.int32)
    del M
    cnt = np.asarray(Mi.sum(axis=0)).ravel()
    C = (Mi.T @ Mi).tocoo()
    del Mi
    m = (C.row != C.col) & (C.data >= min_conteo_par)
    i, j, cij = C.row[m], C.col[m], C.data[m].astype(float)
    del C, m
    if top_k_conf is not None or top_k_lift is not None:
        conf = cij / cnt[i]
        lift = conf / (cnt[j] / n)
        keep = np.zeros(len(i), dtype=bool)
        if top_k_conf:
            keep |= _top_por_grupo(i, conf, top_k_conf)
        if top_k_lift:
            keep |= _top_por_grupo(i, lift, top_k_lift)
        i, j, cij = i[keep], j[keep], cij[keep]
    df = pd.DataFrame({'antecedente': [(int(x),) for x in items[i]],
                       'consecuente': [(int(x),) for x in items[j]],
                       'soporte_antecedente': cnt[i] / n, 'soporte_consecuente': cnt[j] / n,
                       'soporte': cij / n, 'confianza': cij / cnt[i]})
    return agregar_metricas(df)


def construir_base(canastas, soporte_itemsets, min_conteo_par, max_len=3, top_k_conf=None, top_k_lift=None,
                   minador=None):
    # Base = reglas 1->1 (pares) + reglas con antecedente de 2 productos
    pares = reglas_pares(canastas, min_conteo_par, top_k_conf, top_k_lift)
    if minador is None:
        fr = frecuentes_fpgrowth(canastas, soporte_itemsets, max_len=max_len)
    else:
        fr = minador(canastas, soporte_itemsets, max_len)
    multi = generar_reglas(fr, 0.0, solo_consecuente_unitario=True)
    multi = multi[multi['len_antecedente'] >= 2]
    base = pd.concat([pares, multi], ignore_index=True)
    base['consecuente'] = base['consecuente'].map(lambda t: int(t[0]))
    base['n_transacciones'] = np.rint(base['soporte'].to_numpy(float) * len(canastas)).astype(int)
    return base.reset_index(drop=True)


def recomendar_populares(populares, seleccion, k=5):
    sel, out = set(int(x) for x in seleccion), []
    for p in populares:
        if p not in sel:
            out.append(int(p))
            if len(out) == k:
                break
    return out


class Recomendador:
    # Recomienda los consecuentes de las reglas cuyo antecedente está contenido en la selección

    def __init__(self, base, populares):
        self.base = base.reset_index(drop=True)
        self.ant = list(self.base['antecedente'])
        self.len_ant = self.base['len_antecedente'].to_numpy()
        self.ant_set = {int(k): frozenset(self.ant[k]) for k in np.flatnonzero(self.len_ant > 1)}
        self.cons = self.base['consecuente'].to_numpy(np.int64)
        self.sup = self.base['soporte'].to_numpy(float)
        self.conf = self.base['confianza'].to_numpy(float)
        self.lift = self.base['lift'].to_numpy(float)
        self.lev = self.base['leverage'].to_numpy(float)
        idx = defaultdict(list)
        for k, a in enumerate(self.ant):
            for it in a:
                idx[it].append(k)
        self.indice = {it: np.array(v, dtype=np.int64) for it, v in idx.items()}
        self.populares = [int(x) for x in populares]
        self.items_con_reglas = set(self.indice)

    def candidatos(self, sel):
        listas = [self.indice[i] for i in sel if i in self.indice]
        if not listas:
            return np.empty(0, dtype=np.int64)
        idx = np.unique(np.concatenate(listas))
        multi = np.flatnonzero(self.len_ant[idx] > 1)
        if len(multi):
            ok = np.ones(len(idx), dtype=bool)
            for p in multi:
                ok[p] = self.ant_set[idx[p]] <= sel
            idx = idx[ok]
        return idx

    def recomendar(self, seleccion, k=5, min_soporte=0.0, min_confianza=0.0, min_lift=0.0,
                   criterio='confianza', rellenar=False):
        sel = frozenset(int(x) for x in seleccion)
        idx = self.candidatos(sel)
        resultado = []
        if len(idx):
            m = ((self.sup[idx] >= min_soporte) & (self.conf[idx] >= min_confianza)
                 & (self.lift[idx] >= min_lift) & ~np.isin(self.cons[idx], list(sel)))
            idx = idx[m]
        if len(idx):
            clave = {'confianza': self.conf, 'lift': self.lift, 'leverage': self.lev}[criterio]
            idx = idx[np.lexsort((-self.lift[idx], -self.conf[idx], -clave[idx]))]
            _, primeros = np.unique(self.cons[idx], return_index=True)
            for r in idx[np.sort(primeros)][:k]:
                resultado.append({'producto': int(self.cons[r]), 'regla': int(r), 'origen': 'reglas'})
        if rellenar and len(resultado) < k:
            ya = set(sel) | {d['producto'] for d in resultado}
            for p in self.populares:
                if p not in ya:
                    resultado.append({'producto': p, 'regla': None, 'origen': 'popularidad'})
                    ya.add(p)
                    if len(resultado) >= k:
                        break
        return resultado

    def recomendar_ids(self, seleccion, k=5, **kw):
        return [d['producto'] for d in self.recomendar(seleccion, k, **kw)]
