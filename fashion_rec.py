"""Fashion Recommendation Engine - single-file version.
Run:  pip install numpy pandas scipy   then   python fashion_rec.py
Generates synthetic data, cleans it, trains popularity / content / item-CF / ALS / hybrid,
evaluates on a held-out time window, and prints sample recommendations with reasons.
"""
import json, pickle, re, time, itertools
import numpy as np, pandas as pd, scipy.sparse as sp
from pathlib import Path

# ==================== vocab ====================
CATEGORIES = ["Kurta", "Anarkali", "Saree", "Lehenga", "Sherwani", "Nehru Jacket", "Palazzo Set", "Dupatta"]
FABRICS = ["Cotton", "Silk", "Georgette", "Chiffon", "Linen", "Velvet", "Rayon", "Chanderi"]
COLOURS = ["Maroon", "Navy", "Mustard", "Emerald", "Ivory", "Black", "Pink", "Teal", "Rust", "Gold"]
OCCASIONS = ["Casual", "Festive", "Wedding", "Office"]
PRICE_BANDS = ["Budget", "Mid", "Premium"]
SIZES = ["S", "M", "L", "XL", "XXL"]
# every spelling seen in the raw sales feed -> canonical size
SIZE_VARIANTS = {
    "S": ["S", "Small", "36", "s"], "M": ["M", "Medium", "38", "m"], "L": ["L", "Large", "40"],
    "XL": ["XL", "X-Large", "42"], "XXL": ["XXL", "44"],
}
SIZE_MAP = {v.lower(): k for k, vs in SIZE_VARIANTS.items() for v in vs}
EVENTS = ["view", "add_to_cart", "purchase", "return"]
# ==================== data_gen ====================

BASE = pd.Timestamp("2025-01-01")


def generate(out_dir="data/raw", n_users=25000, n_items=1500, days=365, seed=7, mean_interactions=34.0,
             alpha=1.0, beta=6.0):
    rng = np.random.default_rng(seed)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    D = 8
    # ------------------------------------------------------------------ catalogue
    cat = rng.integers(0, len(CATEGORIES), n_items)
    fab = rng.integers(0, len(FABRICS), n_items)
    col = rng.integers(0, len(COLOURS), n_items)
    occ = rng.choice(len(OCCASIONS), n_items, p=[.3, .3, .25, .15])
    pb = rng.choice(3, n_items, p=[.35, .45, .2])
    price = np.round(np.exp(6.9 + .7 * pb + .4 * np.isin(cat, [3, 4]) + rng.normal(0, .2, n_items)), -1)
    E = lambda n: rng.normal(size=(n, D))
    V = (E(8)[cat] + .7 * E(8)[fab] + .7 * E(10)[col] + E(4)[occ] + .5 * E(3)[pb]
         + 1.4 * rng.normal(size=(n_items, D)))            # last term = hidden factor (CF-only signal)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    logpop = rng.normal(0, 1.0, n_items)
    launch = np.where(rng.random(n_items) < .75, 0, rng.integers(60, days - 1, n_items))
    titles = []
    for i in range(n_items):
        parts = []
        if rng.random() < .7: parts.append(COLOURS[col[i]])
        if rng.random() < .6: parts.append(FABRICS[fab[i]])
        titles.append(" ".join(parts + [CATEGORIES[cat[i]]]))
    catalogue = pd.DataFrame({
        "style_id": [f"ST{i:05d}" for i in range(n_items)], "title": titles,
        "category": np.array(CATEGORIES)[cat], "fabric": np.array(FABRICS)[fab], "colour": np.array(COLOURS)[col],
        "occasion": np.array(OCCASIONS)[occ], "price": price, "launch_date": BASE + pd.to_timedelta(launch, "D")})
    catalogue.loc[rng.random(n_items) < .15, "fabric"] = np.nan
    catalogue.loc[rng.random(n_items) < .10, "colour"] = np.nan
    # ------------------------------------------------------------------ users
    K = 8
    cent = rng.normal(size=(K, D))
    U = cent[rng.integers(0, K, n_users)] + .8 * rng.normal(size=(n_users, D))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    join = np.where(rng.random(n_users) < .6, 0, rng.integers(0, days - 2, n_users)).astype(float)
    active = np.clip((days - join) / days, .08, 1)
    rate = rng.lognormal(np.log(mean_interactions) - .45, .95, n_users) * active
    n_int = np.maximum(1, rng.poisson(rate))
    size_pref = rng.choice(5, n_users, p=[.2, .3, .25, .15, .1])
    # ------------------------------------------------------------------ sample interactions
    uu, ii = [], []
    for s in range(0, n_users, 2500):
        sl = slice(s, min(s + 2500, n_users))
        lg = alpha * logpop[None, :] + beta * (U[sl] @ V.T)
        p = np.exp(lg - lg.max(1, keepdims=True)); cum = np.cumsum(p / p.sum(1, keepdims=True), 1)
        for r, u in enumerate(range(sl.start, sl.stop)):
            ii.append(np.minimum(np.searchsorted(cum[r], rng.random(n_int[u])), n_items - 1)); uu.append(np.full(n_int[u], u))
    uu, ii = np.concatenate(uu), np.concatenate(ii)
    aff = (U[uu] * V[ii]).sum(1)
    start = np.maximum(join[uu], launch[ii])
    day = start + rng.random(len(uu)) * (days - start)
    t_view = BASE + pd.to_timedelta(day * 86400, "s")
    # ------------------------------------------------------------------ funnel
    cart = rng.random(len(uu)) < np.clip(.28 + .25 * aff, .05, .8)
    buy = np.where(cart, rng.random(len(uu)) < np.clip(.35 + .3 * aff, .1, .8), rng.random(len(uu)) < .05)
    ret = buy & (rng.random(len(uu)) < .12)
    end = BASE + pd.Timedelta(days=days)
    store_p = rng.dirichlet(np.ones(35) * 2)
    szi = np.where(rng.random(len(uu)) < .7, size_pref[uu], np.clip(size_pref[uu] + rng.choice([-1, 1], len(uu)), 0, 4))
    def mk(mask, ev, dt, store=False):
        n = int(mask.sum())
        d = pd.DataFrame({"user_id": uu[mask].astype(float), "sku": catalogue.style_id.values[ii[mask]],
                          "event_type": ev, "ts": t_view[mask] + dt(n), "size_i": szi[mask], "_row": np.flatnonzero(mask)})
        if store:
            online = rng.random(n) < .35
            d["channel"] = np.where(online, "online", "store")
            d["store_id"] = np.where(online, np.nan, rng.choice(35, n, p=store_p) + 1)
            d["qty"] = rng.choice([1, 2], n, p=[.9, .1])
            d["unit_price"] = catalogue.price.values[ii[mask]] * rng.choice([1, .9, .8], n, p=[.6, .25, .15])
        else:
            d["channel"] = "online"; d["store_id"] = np.nan; d["qty"] = 0; d["unit_price"] = np.nan
        return d
    mn = lambda a, b: (lambda n: pd.to_timedelta(rng.uniform(a, b, n), "m"))
    dd = lambda a, b: (lambda n: pd.to_timedelta(rng.uniform(a, b, n), "D"))
    views = mk(np.ones(len(uu), bool), "view", mn(0, 0))
    carts = mk(cart, "add_to_cart", mn(1, 30))
    buys = mk(buy, "purchase", dd(.05, 3), store=True)
    rets = mk(ret, "return", dd(5, 25), store=True)
    rets["qty"] = 1
    for d in (views, carts, buys, rets):
        d["size_label"] = np.nan
    for d in (carts, buys, rets):
        lab = np.empty(len(d), object)
        for s_i, name in enumerate(SIZES):
            m = (d.size_i == s_i).values; v = np.array(SIZE_VARIANTS[name], object)
            lab[m] = v[rng.integers(0, len(v), m.sum())]
        d["size_label"] = lab
    sales = pd.concat([views, carts, buys, rets], ignore_index=True)
    sales = sales[sales.ts < end].drop(columns=["size_i", "_row"]).sort_values("ts").reset_index(drop=True)
    # dirt
    m = rng.random(len(sales)) < .06
    sales.loc[m, "event_type"] = sales.loc[m, "event_type"].str.upper()
    m = rng.random(len(sales)) < .03
    sales.loc[m, "event_type"] = sales.loc[m, "event_type"].str.title().str.replace("_", " ")
    sales.loc[rng.random(len(sales)) < .003, "user_id"] = np.nan
    sales.insert(0, "txn_id", np.arange(len(sales)))
    # exact duplicate rows (same txn_id) from a double-posting POS feed
    sales = pd.concat([sales, sales.sample(frac=.005, random_state=1)]).sort_values("txn_id", kind="stable").reset_index(drop=True)
    # ------------------------------------------------------------------ inventory with negative stock
    rows = []
    sold = pd.Series(np.exp(logpop)); sold = (sold / sold.mean())
    for st in range(0, 36):                      # 0 = online warehouse
        has = rng.random(n_items) < (.9 if st == 0 else .4)
        idx = np.flatnonzero(has)
        oh = rng.poisson(np.clip(sold.values[idx] * (30 if st == 0 else 6), 0, None))
        neg = rng.random(len(idx)) < .04
        oh = np.where(neg, -rng.integers(1, 16, len(idx)), oh)
        rows.append(pd.DataFrame({"store_id": st, "sku": catalogue.style_id.values[idx], "on_hand": oh}))
    inv = pd.concat(rows, ignore_index=True)
    dead = rng.random(n_items) < .03
    inv.loc[inv.sku.isin(catalogue.style_id[dead]), "on_hand"] = 0
    sales.to_csv(out / "sales_raw.csv", index=False)
    catalogue.to_csv(out / "catalogue_raw.csv", index=False)
    inv.to_csv(out / "inventory_raw.csv", index=False)
    sales.head(5000).to_csv(out.parent / "sample_sales_raw.csv", index=False)
    return len(sales)
# ==================== prep ====================

W_VIEW, W_CART, W_BUY, W_RETURN = 1.0, 3.0, 5.0, -6.0     # implicit-feedback weights (view/cart/buy capped)


def normalise_size(s):
    low = s.str.strip().str.lower()
    mapped = low.map(SIZE_MAP)
    return mapped.where(mapped.notna() | low.isna(), "UNK")


def clean_sales(raw):
    rep = {"rows_raw": len(raw)}
    df = raw.drop_duplicates(); rep["duplicate_rows_dropped"] = rep["rows_raw"] - len(df)
    rep["null_user_rows_dropped"] = int(df.user_id.isna().sum())
    df = df[df.user_id.notna()].copy(); df["user_id"] = df.user_id.astype("int64")
    rep["event_type_variants_before"] = int(df.event_type.nunique())
    df["event_type"] = df.event_type.str.strip().str.lower().str.replace(" ", "_")
    rep["event_type_variants_after"] = int(df.event_type.nunique())
    rep["size_variants_before"] = int(df.size_label.nunique())
    df["size"] = normalise_size(df.size_label)
    rep["size_variants_after"] = int(df["size"].nunique())
    df["ts"] = pd.to_datetime(df.ts)
    df = df.rename(columns={"sku": "style_id"}).drop(columns=["size_label"])
    rep["rows_clean"] = len(df)
    return df.reset_index(drop=True), rep


def clean_catalogue(cat):
    cat = cat.copy(); rep = {}
    for col, vocab in (("colour", COLOURS), ("fabric", FABRICS)):
        miss = cat[col].isna(); rep[f"{col}_missing_before"] = int(miss.sum())
        found = cat.title.str.extract("(" + "|".join(vocab) + ")", flags=re.I, expand=False).str.title()
        cat.loc[miss, col] = found[miss]                       # recover from product title where possible
        cat[col] = cat[col].fillna("Unknown")
        rep[f"{col}_still_unknown"] = int((cat[col] == "Unknown").sum())
    cat["price_band"] = pd.qcut(cat.price, 3, labels=PRICE_BANDS).astype(str)
    cat["launch_date"] = pd.to_datetime(cat.launch_date)
    return cat, rep


def clean_inventory(inv):
    rep = {"negative_stock_rows": int((inv.on_hand < 0).sum()), "rows": len(inv)}
    inv = inv.assign(negative_flag=inv.on_hand < 0, on_hand=inv.on_hand.clip(lower=0))
    stock = inv.groupby("sku").on_hand.sum().rename("stock")
    return stock, rep


def aggregate(df):
    """One row per (user, item) with event counts, capped weighted score and strongest level."""
    g = df.groupby(["u", "i", "event_type"]).size().unstack(fill_value=0)
    for c in EVENTS:
        if c not in g: g[c] = 0
    g = g.reset_index()
    g["w"] = (W_VIEW * np.minimum(g["view"], 3) + W_CART * np.minimum(g["add_to_cart"], 2)
              + W_BUY * np.minimum(g["purchase"], 2) + W_RETURN * g["return"])
    g["level"] = np.where(g.purchase > 0, 2, np.where(g.add_to_cart > 0, 1, 0))
    return g


def to_csr(g, shape, values=None, mask=None):
    if mask is not None: g = g[mask]
    v = np.ones(len(g)) if values is None else g[values].values
    return sp.csr_matrix((v, (g.u.values, g.i.values)), shape=shape, dtype=np.float32)
# ==================== models ====================

CHUNK = 3000


def _z(M):
    return ((M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-9)).astype(np.float32)


def _l2(M):
    M = sp.csr_matrix(M, dtype=np.float32)
    n = np.sqrt(np.asarray(M.multiply(M).sum(1)).ravel()); n[n == 0] = 1
    return sp.diags(1 / n) @ M


def _log(R):
    R = R.copy(); R.data = np.log1p(R.data); return R


# ----------------------------------------------------------------------------- features
FEATURE_WEIGHTS = {"category": 1.0, "fabric": 1.0, "colour": .7, "occasion": 1.0, "price_band": .5}


def build_item_features(cat):
    blocks, names = [], []
    for col, w in FEATURE_WEIGHTS.items():
        d = pd.get_dummies(cat[col].astype(str), prefix=col, dtype=np.float32)
        blocks.append(d.values * w); names += list(d.columns)
    F = _l2(np.hstack(blocks)); return F, np.array(names)


# ----------------------------------------------------------------------------- models
class Popularity:
    def fit(self, R):
        self.pop = np.asarray(R.sum(0)).ravel().astype(np.float32); return self
    def score(self, users):
        return np.broadcast_to(self.pop[None, :], (len(users), len(self.pop))).copy()


class ContentBased:
    def __init__(self, F): self.F = F
    def fit(self, R): self.Rl = _log(R); return self
    def profile(self, users): return _l2(self.Rl[users] @ self.F)
    def score(self, users):
        return np.asarray((self.profile(users) @ self.F.T).todense(), dtype=np.float32)


class ItemCF:
    def __init__(self, k=80, shrink=1.0): self.k, self.shrink = k, shrink
    def fit(self, R):
        self.Rl = _log(R)
        C = (self.Rl.T @ self.Rl).toarray()
        n = np.sqrt(np.diag(C))
        S = C / (np.outer(n, n) + self.shrink); np.fill_diagonal(S, 0)
        drop = np.argpartition(-S, self.k, axis=1)[:, self.k:]       # keep k nearest neighbours per item
        np.put_along_axis(S, drop, 0, axis=1)
        self.S = S.astype(np.float32); return self
    def score(self, users):
        return np.asarray(self.Rl[users] @ self.S, dtype=np.float32)


class ALS:
    """Implicit-feedback ALS (Hu, Koren, Volinsky 2008): confidence c = 1 + alpha * r, preference p = 1[r>0]."""
    def __init__(self, factors=32, reg=.1, alpha=8., iters=10, seed=0):
        self.f, self.reg, self.alpha, self.iters, self.seed = factors, reg, alpha, iters, seed
    def _solve(self, M, Y):
        YtY = Y.T @ Y + self.reg * np.eye(self.f, dtype=np.float32)
        X = np.zeros((M.shape[0], self.f), np.float32)
        ip, ix, dt = M.indptr, M.indices, M.data
        for u in range(M.shape[0]):
            s, e = ip[u], ip[u + 1]
            if s == e: continue
            Yu = Y[ix[s:e]]; c = self.alpha * dt[s:e]
            X[u] = np.linalg.solve(YtY + (Yu.T * c) @ Yu, (Yu.T * (1 + c)).sum(1))
        return X
    def fit(self, R):
        R = R.tocsr(); RT = R.T.tocsr()
        rng = np.random.default_rng(self.seed)
        self.Y = (rng.normal(size=(R.shape[1], self.f)) * .1).astype(np.float32)
        for _ in range(self.iters):
            self.X = self._solve(R, self.Y); self.Y = self._solve(RT, self.X)
        return self
    def score(self, users): return self.X[users] @ self.Y.T


class Hybrid:
    """Blend of z-scored model scores. Weights depend on how much history a user has; items with too
    little interaction history (new / long-tail) get their CF signal replaced by the content signal plus a boost."""
    def __init__(self, pop, cb, icf, als, R, min_support=5, new_boost=.8, warm_min=5):
        self.pop, self.cb, self.icf, self.als = pop, cb, icf, als
        self.nnz = np.diff(R.indptr)
        self.item_count = np.diff(R.tocsc().indptr)
        self.thin = self.item_count < min_support
        self.new_boost, self.warm_min = new_boost, warm_min
        lp = np.log1p(pop.pop); self.zlogpop = ((lp - lp.mean()) / lp.std()).astype(np.float32)

    def base_score(self, users):
        users = np.asarray(users); out = np.empty((len(users), len(self.thin)), np.float32)
        for s in range(0, len(users), CHUNK):
            u = users[s:s + CHUNK]; nnz = self.nnz[u][:, None]
            Zp, Zc, Zi, Za = (_z(m.score(u)) for m in (self.pop, self.cb, self.icf, self.als))
            Zi[:, self.thin] = Zc[:, self.thin]; Za[:, self.thin] = Zc[:, self.thin]
            cold, warm = nnz == 0, nnz >= self.warm_min
            wp = np.where(cold, 1., np.where(warm, .05, .20)); wc = np.where(cold, 0., np.where(warm, .20, .40))
            wi = np.where(cold, 0., np.where(warm, .25, .15)); wa = np.where(cold, 0., np.where(warm, .50, .25))
            S = wp * Zp + wc * Zc + wi * Zi + wa * Za
            S[:, self.thin] += self.new_boost
            out[s:s + len(u)] = S
        return out

    def finalize(self, S, users, pop_penalty=0.):
        has = (self.nnz[np.asarray(users)] > 0)[:, None]
        return S - pop_penalty * has * self.zlogpop[None, :]


# ----------------------------------------------------------------------------- ranking
def score_chunked(model, users):
    return np.vstack([model.score(users[s:s + CHUNK]) for s in range(0, len(users), CHUNK)])


def topk(S, users, seen, k=10, sim=None, lam=0., eligible=None, pool=40):
    """Mask seen/ineligible items, take top-k; optional MMR re-ranking (lam=0 -> pure relevance)."""
    S = S.copy()
    S[seen[users].toarray() > 0] = -np.inf
    if eligible is not None: S[:, ~eligible] = -np.inf
    if lam <= 0 or sim is None:
        idx = np.argpartition(-S, k, axis=1)[:, :k]
        return np.take_along_axis(idx, np.argsort(-np.take_along_axis(S, idx, 1), axis=1), 1)
    cand = np.argpartition(-S, pool, axis=1)[:, :pool]
    cs = np.take_along_axis(S, cand, 1)
    cs = (cs - cs.min(1, keepdims=True)) / (cs.max(1, keepdims=True) - cs.min(1, keepdims=True) + 1e-9)
    out = np.zeros((len(S), k), int)
    for r in range(len(S)):
        c = cand[r]; sm = sim[np.ix_(c, c)]; mx = np.zeros(pool); chosen = []
        for _ in range(k):
            v = (1 - lam) * cs[r] - lam * mx; v[chosen] = -np.inf
            j = int(np.argmax(v)); chosen.append(j); mx = np.maximum(mx, sm[j])
        out[r] = c[chosen]
    return out


class Bundle:
    """Everything the serving layer needs (fitted models, matrices, catalogue)."""
    pass
# ==================== metrics ====================


def per_user_metrics(recs, rel, sim, thin=None):
    """recs: (n,k) item ids; rel: (n,items) binary relevance; sim: item-item content cosine (diag = 1)."""
    n, k = recs.shape
    hits = np.take_along_axis(rel, recs, 1).astype(np.float32)
    nrel = rel.sum(1)
    disc = 1 / np.log2(np.arange(2, k + 2)); cum = np.cumsum(disc)
    S = sim[recs[:, :, None], recs[:, None, :]]
    d = pd.DataFrame({
        "precision": hits.sum(1) / k, "recall": hits.sum(1) / nrel,
        "ndcg": (hits * disc).sum(1) / cum[np.minimum(nrel, k).astype(int) - 1],
        "diversity": 1 - (S.sum((1, 2)) - k) / (k * (k - 1))})
    if thin is not None:
        d["new_hits"] = (hits * thin[recs]).sum(1); d["new_rel"] = rel[:, thin].sum(1)
    return d


def summarise(d, recs, n_items):
    out = {"precision@10": d.precision.mean(), "recall@10": d.recall.mean(), "ndcg@10": d.ndcg.mean(),
           "coverage": len(np.unique(recs)) / n_items, "diversity": d.diversity.mean()}
    if "new_rel" in d and d.new_rel.sum() > 0:
        out["new_item_recall"] = d.new_hits.sum() / d.new_rel.sum()
    return out
# ==================== serve ====================

MODELS = ["hybrid", "popularity", "content_based", "item_item_cf", "als"]
VERB = {3: "bought", 2: "added to your cart", 1: "viewed"}


class Recommender:
    def __init__(self, bundle):
        self.b = bundle
        self.uidx = {int(u): i for i, u in enumerate(bundle.user_ids)}
        self.anon = len(bundle.user_ids)                                   # empty row for unknown users
        c = bundle.cat
        self.title = np.array([" ".join(x for x in (a, f, k) if x != "Unknown") for a, f, k in zip(c.colour, c.fabric, c.category)])
        launched = bundle.cat.launch_day.values
        self.eligible = (bundle.cat.stock.values > 0) & (launched <= bundle.cutoff_day if bundle.cutoff_day < 1e8 else True)

    @classmethod
    def load(cls, path="artifacts/bundle.pkl"):
        with open(path, "rb") as f: return cls(pickle.load(f))

    def user_type(self, user_id):
        nnz = int(self.b.hybrid.nnz[self.uidx.get(int(user_id), self.anon)])
        return "cold" if nnz == 0 else "light" if nnz < 10 else "heavy" if nnz > 40 else "regular"

    def recommend(self, user_id, k=10, model="hybrid", pop_penalty=None, mmr_lambda=None):
        b = self.b; u = self.uidx.get(int(user_id), self.anon); us = np.array([u])
        pp = b.best["pop_penalty"] if pop_penalty is None else pop_penalty
        lam = b.best["mmr_lambda"] if mmr_lambda is None else mmr_lambda
        if model == "hybrid": S = b.hybrid.finalize(b.hybrid.base_score(us), us, pp)
        else: S = {"popularity": b.pop, "content_based": b.cb, "item_item_cf": b.icf, "als": b.als}[model].score(us)
        items = topk(S, us, b.seen, k, b.sim, lam if model == "hybrid" else 0, self.eligible)[0]
        return [self._card(u, int(j), r + 1, model) for r, j in enumerate(items)]

    # ------------------------------------------------------------------ explanations
    def _card(self, u, j, rank, model):
        c = self.b.cat.iloc[j]
        return {"rank": rank, "style_id": c.style_id, "title": c.title, "category": c.category, "fabric": c.fabric,
                "colour": c.colour, "occasion": c.occasion, "price": float(c.price), "reason": self._why(u, j)}

    def _why(self, u, j):
        b = self.b
        if b.hybrid.nnz[u] == 0:
            return "New arrival to the store" if b.hybrid.thin[j] else "Trending across our stores right now"
        row = b.L[u]; hist, lvl = row.indices, row.data
        sims = b.icf.S[hist, j] * (1 + lvl)
        t = int(np.argmax(sims)) if len(hist) else -1
        prof = b.cb.profile(np.array([u])).toarray().ravel() * (b.F[j].toarray().ravel() > 0)
        top = [n.split("_", 1)[1] for n in b.feature_names[np.argsort(-prof)[:2]] if prof[np.argsort(-prof)[0]] > 0 and "Unknown" not in n]
        if b.hybrid.thin[j]:
            return "New arrival that matches your taste for " + " & ".join(top) if top else "New arrival we think you'll like"
        if t >= 0 and sims[t] > .1 * (1 + lvl[t]):
            return f"Because you {VERB[int(lvl[t])]} “{self.title[hist[t]]}”"
        if top: return "Matches your taste for " + " & ".join(top)
        return "Popular with shoppers like you"
# ==================== train ====================

HP = dict(als_factors=16, als_reg=.1, als_alpha=3., als_iters=10, icf_k=80, min_support=5, new_boost=.8)
VAL_DAY, TEST_DAY = 245, 305           # train < day 245 | validation 245-305 | test >= 305 (final model refits on < 305)
K = 10


def load_clean(data_dir="data", regenerate=False):
    raw = Path(data_dir) / "raw"
    if regenerate or not (raw / "sales_raw.csv").exists():
        generate(raw)
    sales, r1 = clean_sales(pd.read_csv(raw / "sales_raw.csv"))
    cat, r2 = clean_catalogue(pd.read_csv(raw / "catalogue_raw.csv"))
    stock, r3 = clean_inventory(pd.read_csv(raw / "inventory_raw.csv"))
    cat = cat.join(stock, on="style_id"); cat["stock"] = cat.stock.fillna(0).astype(int)
    user_ids = np.sort(sales.user_id.unique())
    sales["u"] = np.searchsorted(user_ids, sales.user_id.values)
    sales["i"] = sales.style_id.map({s: i for i, s in enumerate(cat.style_id)}).astype(int)
    sales["day"] = (sales.ts - BASE).dt.total_seconds() / 86400
    return sales, cat, user_ids, dict(sales=r1, catalogue=r2, inventory=r3)


def fit_bundle(sales, cat, user_ids, cutoff_day, hp=HP):
    b = Bundle(); b.hp = hp; b.cutoff_day = cutoff_day
    nu, ni = len(user_ids) + 1, len(cat)                     # last user row = anonymous / unknown user
    tr = sales[sales.day < cutoff_day]
    g = aggregate(tr)
    b.R = to_csr(g, (nu, ni), "w", g.w > 0)
    b.seen = to_csr(g, (nu, ni))
    b.L = to_csr(g, (nu, ni), "level", None); b.L.data += 1
    b.cat, b.user_ids = cat.reset_index(drop=True), user_ids
    b.F, b.feature_names = build_item_features(cat)
    b.sim = np.asarray((b.F @ b.F.T).todense(), dtype=np.float32)
    b.pop = Popularity().fit(b.R); b.cb = ContentBased(b.F).fit(b.R)
    b.icf = ItemCF(hp["icf_k"]).fit(b.R)
    b.als = ALS(hp["als_factors"], hp["als_reg"], hp["als_alpha"], hp["als_iters"]).fit(b.R)
    b.hybrid = Hybrid(b.pop, b.cb, b.icf, b.als, b.R, hp["min_support"], hp["new_boost"])
    return b


def make_eval(sales, b, start_day, end_day=1e9):
    w = sales[(sales.day >= start_day) & (sales.day < end_day)]
    g = aggregate(w)
    g = g[(g.w >= 3) & (g.add_to_cart + g.purchase > 0) & (g["return"] == 0)]
    g = g[np.asarray(b.seen[g.u.values, g.i.values]).ravel() == 0]        # only items new to the user
    users = np.unique(g.u.values)
    rel = np.zeros((len(users), b.R.shape[1]), np.float32)
    rel[np.searchsorted(users, g.u.values), g.i.values] = 1
    nnz = b.hybrid.nnz[users]; warm = nnz[nnz > 0]
    q1, q3 = np.percentile(warm, [25, 75])
    grp = np.where(nnz == 0, "cold", np.where(nnz >= q3, "heavy", np.where(nnz <= q1, "light", "mid")))
    return dict(users=users, rel=rel, group=grp, n=len(users))


def _eval_recs(b, ev, recs):
    d = per_user_metrics(recs, ev["rel"], b.sim, b.hybrid.thin)
    return d, summarise(d, recs, len(b.cat))


def compare_models(b, ev):
    u = ev["users"]; rows, per_group = {}, []
    S0 = b.hybrid.base_score(u)
    scorers = {"popularity": lambda: score_chunked(b.pop, u), "content_based": lambda: score_chunked(b.cb, u),
               "item_item_cf": lambda: score_chunked(b.icf, u), "als": lambda: score_chunked(b.als, u),
               "hybrid (no re-rank)": lambda: S0}
    for name, fn in scorers.items():
        recs = topk(fn(), u, b.seen, K); d, s = _eval_recs(b, ev, recs); rows[name] = s
        for gname in ("heavy", "mid", "light", "cold"):
            m = ev["group"] == gname
            if m.sum(): per_group.append({"model": name, "group": gname, "users": int(m.sum()),
                                          **summarise(d[m], recs[m], len(b.cat))})
    return pd.DataFrame(rows).T, pd.DataFrame(per_group), S0


def sweep(b, ev, S0, pens=(0, .05, .1, .2, .3), lams=(0, .15, .3, .5)):
    u = ev["users"]; out = []
    for pen, lam in itertools.product(pens, lams):
        recs = topk(b.hybrid.finalize(S0, u, pen), u, b.seen, K, b.sim, lam)
        _, s = _eval_recs(b, ev, recs); out.append({"pop_penalty": pen, "mmr_lambda": lam, **s})
    return pd.DataFrame(out)


def choose(sw, min_cov=.40, margin=.02, min_div=.70):
    """Max NDCG subject to coverage >= 40% (+2pt safety margin) and intra-list diversity >= 0.70.
    This is the accuracy-vs-diversity trade-off made explicit: we give up as little NDCG as possible for a diverse, wide-coverage list."""
    ok = sw[(sw.coverage >= min_cov + margin) & (sw.diversity >= min_div)]
    if not len(ok): ok = sw[sw.coverage >= min_cov + margin]
    return (ok if len(ok) else sw).sort_values("ndcg@10", ascending=False).iloc[0]


def tune(sales, cat, user_ids):
    b = fit_bundle(sales, cat, user_ids, VAL_DAY); ev = make_eval(sales, b, VAL_DAY, TEST_DAY)
    tbl, _, S0 = compare_models(b, ev); sw = sweep(b, ev, S0)
    return b, ev, tbl, sw, choose(sw)


def final_eval(sales, cat, user_ids, best):
    b = fit_bundle(sales, cat, user_ids, TEST_DAY); ev = make_eval(sales, b, TEST_DAY)
    tbl, grp, S0 = compare_models(b, ev)
    u = ev["users"]
    recs = topk(b.hybrid.finalize(S0, u, best.pop_penalty), u, b.seen, K, b.sim, best.mmr_lambda)
    d, s = _eval_recs(b, ev, recs); tbl.loc["hybrid (tuned: pop-penalty + MMR)"] = s
    for gname in ("heavy", "mid", "light", "cold"):
        m = ev["group"] == gname
        if m.sum(): grp = pd.concat([grp, pd.DataFrame([{"model": "hybrid (tuned: pop-penalty + MMR)", "group": gname,
                                     "users": int(m.sum()), **summarise(d[m], recs[m], len(b.cat))}])])
    return b, ev, tbl, grp, sweep(b, ev, S0)


def ab_plan(tbl, sales, path, sessions_per_day=20000, realisation=.4):
    pop, best = tbl.loc["popularity"], tbl.iloc[-1]
    lift = best["precision@10"] / pop["precision@10"] - 1
    ev_ = sales.event_type.value_counts(); p0 = ev_["add_to_cart"] / ev_["view"]
    up = min(realisation * lift, .10); dlt = p0 * up          # plan for a conservative <=10% relative lift
    n = int(np.ceil(2 * (1.96 + .84) ** 2 * p0 * (1 - p0) / dlt ** 2))
    txt = f"""# Offline -> online A/B test plan

**Hypothesis.** Replacing the shared popular-items shelf with personalised top-10 recommendations raises add-to-cart rate per recommendation impression.

**Design.** User-level randomisation (50/50, hash of user_id), control = popularity shelf, treatment = tuned hybrid. Online channel first (clean attribution); stores get the same recs via the associate app in phase 2 with store-level holdout. Run >= 2 full weeks to cover weekday/weekend and a festive-sales cycle.

**Metrics.** Primary: add-to-cart rate per impression. Secondary: purchase conversion, revenue/user, catalogue coverage of impressions, return rate (guardrail), page latency p95 < 150 ms (guardrail).

**Expected uplift (estimate).** Offline precision@10 lift of hybrid over popularity is {lift:+.0%}. Offline gains usually shrink online (position bias, stock-outs, feedback loops), so I assume {realisation:.0%} realisation -> the raw estimate is far above any realistic online gain because the synthetic taste signal is strong, so the test is sized for a conservative **{up:+.0%} relative uplift** in the primary metric (the real number must come from this experiment).

**Sample size.** Baseline add-to-cart rate per view in the data p0 = {p0:.3f}. To detect an absolute lift of {dlt:.4f} (alpha 0.05 two-sided, power 80%): **~{n:,} impressions per arm**. At an assumed {sessions_per_day:,} sessions/day (10 impressions each, 50/50 split) this reads out in ~{max(1, int(np.ceil(n / (sessions_per_day * 10 / 2))))} day(s), so the 2-week minimum is the binding constraint, not power.

**Risks / caveats.** Synthetic-data estimate; replay offline on real logs first; check cannibalisation of store sales; monitor stock-outs (recs are already filtered to in-stock items); ramp 5% -> 25% -> 50%.
"""
    Path(path).write_text(txt); return dict(lift=lift, uplift=up, p0=p0, n_per_arm=n)


def export(b, path="artifacts/bundle.pkl"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    b.cat["launch_day"] = (b.cat.launch_date - BASE).dt.days
    with open(path, "wb") as f: pickle.dump(b, f, protocol=4)


def main(data_dir="data", out_dir="results", regenerate=False):
    out = Path(out_dir); out.mkdir(exist_ok=True); t0 = time.time()
    sales, cat, user_ids, rep = load_clean(data_dir, regenerate)
    (out / "cleaning_report.json").write_text(json.dumps(rep, indent=2)); print("clean ok", rep, f"{time.time()-t0:.0f}s")
    _, _, vtbl, vsw, best = tune(sales, cat, user_ids); vsw.to_csv(out / "tradeoff_validation.csv", index=False)
    print("chosen on validation:", dict(best), f"{time.time()-t0:.0f}s")
    b, ev, tbl, grp, tsw = final_eval(sales, cat, user_ids, best)
    tbl.round(4).to_csv(out / "evaluation_table.csv"); grp.round(4).to_csv(out / "evaluation_by_user_group.csv", index=False)
    tsw.round(4).to_csv(out / "tradeoff_test.csv", index=False)
    print(tbl.round(4)); print(grp.round(4).to_string()); print("A/B:", ab_plan(tbl, sales, out / "ab_test_plan.md"))
    pop, hy = tbl.loc["popularity", "ndcg@10"], tbl.iloc[-1]
    print(f"SUCCESS CHECK  NDCG lift vs popularity: {hy['ndcg@10']/pop-1:+.1%} (need >=25%)  coverage: {hy['coverage']:.1%} (need >=40%)")
    final = fit_bundle(sales, cat, user_ids, 1e9)             # refit on all data for serving
    final.best = dict(pop_penalty=float(best.pop_penalty), mmr_lambda=float(best.mmr_lambda))
    export(final); print("done", f"{time.time()-t0:.0f}s")




def demo():
    rec = Recommender.load()
    nnz = rec.b.hybrid.nnz[:-1]
    for name, mask in [("heavy", nnz > 40), ("light", (nnz > 0) & (nnz < 10)), ("cold-start", nnz == 0)]:
        uid = int(rec.b.user_ids[np.flatnonzero(mask)[3]])
        print(f"\n--- {name} shopper {uid}")
        for r in rec.recommend(uid, 5): print(f"{r['rank']}. {r['title']:<28} {r['reason']}")


if __name__ == "__main__":
    main()
    demo()
