"""
Minimal standalone ASTRA implementation.

ASTRA classifies each galaxy by its local cosmic-web environment
(void / sheet / filament / knot) using a Delaunay triangulation over the
combined data + random catalog:

  r = (ndata_neighbours - nrand_neighbours) / (ndata_neighbours + nrand_neighbours)

  r ∈ [−1, −0.9]  → void
  r ∈ (−0.9,  0]  → sheet
  r ∈ (  0,  0.9] → filament
  r ∈ (0.9,   1]  → knot

Galaxies are then grouped into quantiles of r (Q1 = most underdense,
Q4 = most overdense).  The same bin edges are applied to the randoms so both
populations are split at identical density thresholds.
"""

import numpy as np
import pandas as pd
from itertools import combinations
from scipy.spatial import Delaunay
from tqdm import tqdm


class AstraSplit:

    def __init__(self):
        pass

    # ── random generation ──────────────────────────────────────────────────────

    def generate_uniform_randoms(self, positions, n_factor=1, seed=None):
        """
        Draw uniform randoms spanning the same bounding box as positions.

        Bounds are derived from the per-axis min/max of the input, so this
        works for any geometry (cubic, rectangular, offset, non-centred).

        Parameters
        ----------
        positions : (N, 3) array
        n_factor  : int         — randoms = n_factor × len(positions)
        seed      : int or None

        Returns
        -------
        random_positions : (n_factor*N, 3) array
        """
        lo     = positions.min(axis=0)
        hi     = positions.max(axis=0)
        n_rand = n_factor * len(positions)
        rng    = np.random.default_rng(seed)
        return rng.uniform(low=lo, high=hi, size=(n_rand, 3))

    # ── dataframe builder ──────────────────────────────────────────────────────

    def build_dataframe(self, positions, random_positions):
        """
        Concatenate data and randoms into a single ASTRA-format DataFrame.

        Data rows get RANDITER = -1; random rows get RANDITER = 0.
        TARGETID is a contiguous integer index: 0…n_data−1 for data,
        n_data…n_data+n_rand−1 for randoms — used later to map rows back
        to the original position arrays.
        """
        n_data = len(positions)
        n_rand = len(random_positions)

        df_data = pd.DataFrame({
            'TARGETID': np.arange(n_data),
            'XCART':    positions[:, 0],
            'YCART':    positions[:, 1],
            'ZCART':    positions[:, 2],
            'RANDITER': -1,
        })
        df_rand = pd.DataFrame({
            'TARGETID': np.arange(n_data, n_data + n_rand),
            'XCART':    random_positions[:, 0],
            'YCART':    random_positions[:, 1],
            'ZCART':    random_positions[:, 2],
            'RANDITER': 0,
        })
        return pd.concat([df_data, df_rand], ignore_index=True)

    # ── core ASTRA algorithm ───────────────────────────────────────────────────

    def classify(self, df):
        """
        Run the ASTRA classification on a combined data+random DataFrame.

        1. Build a Delaunay triangulation over all points.
        2. For each point i, count data neighbours (ndata) and random
           neighbours (nrand) from the triangulation graph.
        3. For data points: r = (ndata - nrand) / (ndata + nrand).

        Parameters
        ----------
        df : DataFrame with columns XCART, YCART, ZCART, TARGETID, RANDITER

        Returns
        -------
        class_rows : list of (TARGETID, RANDITER, ISDATA, NDATA, NRAND) tuples
        """
        coords    = df[['XCART', 'YCART', 'ZCART']].values
        targetids = df['TARGETID'].values
        is_data   = (df['RANDITER'] == -1).values
        n_points  = len(coords)

        if n_points < 4:
            raise ValueError('Need at least 4 points for Delaunay triangulation.')

        print(f'Delaunay triangulation on {n_points:,} points ...')
        tri       = Delaunay(coords)
        neighbors = {i: set() for i in range(n_points)}

        for simplex in tqdm(tri.simplices, desc='Building neighbour graph'):
            for i, j in combinations(simplex, 2):
                neighbors[i].add(j)
                neighbors[j].add(i)

        print('Computing local densities ...')
        class_rows = []
        for i, nbrs in tqdm(neighbors.items(), desc='Classifying points'):
            nbr_list = list(nbrs)
            ndata    = int(np.sum(is_data[nbr_list]))
            nrand    = len(nbr_list) - ndata
            class_rows.append((
                int(targetids[i]),
                0,
                bool(is_data[i]),
                ndata,
                nrand,
            ))

        return class_rows

    def classify_fast(self, df):
        """
        Vectorised version of classify() for large catalogs (millions of
        points).  Identical results; avoids the per-point Python loop and
        dict-of-sets neighbour graph, which do not scale past ~10^6 points.

        1. Delaunay triangulation over all points.
        2. Expand every tetrahedron into its 6 edges, deduplicate them by
           encoding (i, j) with i < j into a single int64.
        3. Count data / random neighbours per point with np.bincount.

        Returns
        -------
        df_class : DataFrame with columns TARGETID, RANDITER, ISDATA,
                   NDATA, NRAND — same content as classify(), as a frame.
        """
        coords    = df[['XCART', 'YCART', 'ZCART']].values
        targetids = df['TARGETID'].values
        is_data   = (df['RANDITER'] == -1).values
        n_points  = len(coords)

        if n_points < 4:
            raise ValueError('Need at least 4 points for Delaunay triangulation.')

        print(f'Delaunay triangulation on {n_points:,} points ...')
        tri  = Delaunay(coords)
        simp = tri.simplices                      # (M, 4) int32

        print('Extracting unique edges ...')
        edges = np.vstack([simp[:, [a, b]] for a, b in combinations(range(4), 2)])
        i = edges.min(axis=1).astype(np.int64)
        j = edges.max(axis=1).astype(np.int64)
        del edges
        code = np.unique(i * n_points + j)        # dedupe undirected edges
        del i, j
        i = code // n_points
        j = code % n_points
        del code

        print(f'Counting neighbours over {len(i):,} edges ...')
        w = is_data.astype(np.float64)
        ndata = (np.bincount(i, weights=w[j], minlength=n_points) +
                 np.bincount(j, weights=w[i], minlength=n_points)).astype(np.int64)
        ntot  = (np.bincount(i, minlength=n_points) +
                 np.bincount(j, minlength=n_points)).astype(np.int64)

        return pd.DataFrame({
            'TARGETID': targetids.astype(np.int64),
            'RANDITER': 0,
            'ISDATA':   is_data,
            'NDATA':    ndata,
            'NRAND':    ntot - ndata,
        })

    # ── quantile assignment ────────────────────────────────────────────────────

    def assign_quantiles(self, class_rows, n_quantiles=4):
        """
        Convert classification rows to a DataFrame with QUARTILE labels.

        Data and randoms are each split independently with pd.qcut so
        both populations have approximately equal counts per quantile.
        Q1 = most underdense, Q{n_quantiles} = most overdense in each.

        Parameters
        ----------
        class_rows  : output of classify() (list of tuples) or
                      classify_fast() (DataFrame)
        n_quantiles : int

        Returns
        -------
        df : DataFrame with columns
             TARGETID, ISDATA_BOOL, r, QUARTILE
             QUARTILE is 1…n_quantiles for both data and randoms.
        """
        if isinstance(class_rows, pd.DataFrame):
            df = class_rows.copy()
        else:
            df = pd.DataFrame(
                class_rows,
                columns=['TARGETID', 'RANDITER', 'ISDATA', 'NDATA', 'NRAND'],
            )
        df['ISDATA_BOOL'] = df['ISDATA'].astype(bool)
        df['r'] = np.where(
            (df['NDATA'] + df['NRAND']) > 0,
            (df['NDATA'] - df['NRAND']) / (df['NDATA'] + df['NRAND']),
            np.nan,
        )

        data_mask = df['ISDATA_BOOL'] & df['r'].notna()
        rand_mask = ~df['ISDATA_BOOL'] & df['r'].notna()
        labels    = list(range(1, n_quantiles + 1))

        # Each population split independently so both have equal counts per quantile
        df['QUARTILE'] = np.nan
        df.loc[data_mask, 'QUARTILE'] = pd.qcut(
            df.loc[data_mask, 'r'], n_quantiles,
            labels=labels, duplicates='drop',
        ).astype(float)

        df.loc[rand_mask, 'QUARTILE'] = pd.qcut(
            df.loc[rand_mask, 'r'], n_quantiles,
            labels=labels, duplicates='drop',
        ).astype(float)

        print('Quantile distribution (data):')
        print(df.loc[data_mask, 'QUARTILE'].value_counts().sort_index().to_string())
        print('Quantile distribution (randoms):')
        print(df.loc[rand_mask, 'QUARTILE'].value_counts().sort_index().to_string())

        return df
