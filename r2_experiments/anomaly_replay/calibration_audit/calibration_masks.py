"""Chronological detector calibration masks with auditable raw-history separation.

The caller supplies raw_support_start/end after inspecting the actual feature source.
This module does not guess a purge from sequence length alone. Bounds refer to
raw observation dependencies of each complete anomaly feature row, including
forecast inputs with nested lag/rolling features and the current observed meter.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import math
import numpy as np
import pandas as pd

@dataclass
class CalibrationSplit:
    fit_mask: np.ndarray
    threshold_mask: np.ndarray
    purged_mask: np.ndarray
    assignments: pd.DataFrame
    boundaries: pd.DataFrame
    fault_coverage: pd.DataFrame


def _timestamps(series):
    return pd.to_datetime(series)


def make_calibration_split(frame: pd.DataFrame, fraction: float=.6, *,
                           purge: bool=False, building_col='building_id',
                           time_col='timestamp', label_col='anomaly',
                           fault_col='atype', support_start_col='raw_support_start',
                           support_end_col='raw_support_end') -> CalibrationSplit:
    """Return masks aligned with the original positional order of ``frame``.

    Nominal cut k=floor(fraction*n) is defined before any rows are purged.
    A purged split drops equal numbers from the ends of the two adjacent phases,
    with a one-row excess on the threshold side if an odd total is needed.
    It searches the minimal such total that makes *all* fitting raw-time bounds
    precede every threshold raw-time bound in the same building.
    """
    if not 0 < fraction < 1: raise ValueError('fraction must lie strictly between0 and1')
    required=[building_col,time_col,label_col,fault_col]
    if purge: required += [support_start_col,support_end_col]
    missing=[c for c in required if c not in frame]
    if missing: raise ValueError(f'Missing required columns: {missing}')
    f=frame.copy().reset_index(drop=True)
    f['_position']=np.arange(len(f))
    f[time_col]=_timestamps(f[time_col])
    if f.duplicated([building_col,time_col]).any():
        raise ValueError('Building/time keys must be unique')
    has_support=support_start_col in f and support_end_col in f
    if has_support:
        f[support_start_col]=_timestamps(f[support_start_col])
        f[support_end_col]=_timestamps(f[support_end_col])
        if f[[support_start_col,support_end_col]].isna().any().any():
            raise ValueError('Raw support timestamps must be complete')
        if not (f[support_start_col] <= f[support_end_col]).all():
            raise ValueError('Raw support start exceeds end')
        if not (f[support_end_col] <= f[time_col]).all():
            raise ValueError('A feature row consumes a future raw observation')
    roles=np.full(len(f),'purged',dtype=object)
    boundaries=[]
    for bid,g in f.groupby(building_col,sort=True):
        g=g.sort_values(time_col,kind='stable').reset_index(drop=True)
        n=len(g);k=int(math.floor(fraction*n))
        if k==0 or k==n: raise ValueError(f'Empty nominal phase for building{bid}')
        left=right=0
        if purge:
            found=False
            for total in range(n-1):
                dl=total//2;dr=total-dl
                if k-dl<1 or k+dr>=n: continue
                fi=g.iloc[:k-dl];th=g.iloc[k+dr:]
                if fi[support_end_col].max() < th[support_start_col].min():
                    left,right=dl,dr;found=True;break
            if not found:
                raise ValueError(f'Cannot create nonempty raw-history-disjoint phases for building{bid}')
        fi=g.iloc[:k-left];th=g.iloc[k+right:]
        roles[fi['_position'].to_numpy(int)]='fit'
        roles[th['_position'].to_numpy(int)]='threshold'
        row={'building_id':bid,'nominal_fraction':fraction,'purged':purge,
             'nominal_rows':n,'nominal_fit_rows':k,'nominal_threshold_rows':n-k,
             'fit_rows':len(fi),'threshold_rows':len(th),'purged_fit_side_rows':left,
             'purged_threshold_side_rows':right,'purged_rows':left+right,
             'fit_first_target':fi[time_col].min(),'fit_last_target':fi[time_col].max(),
             'threshold_first_target':th[time_col].min(),'threshold_last_target':th[time_col].max(),
             'target_boundary_gap_hours':(th[time_col].min()-fi[time_col].max()).total_seconds()/3600}
        if has_support:
            row.update({'fit_last_raw_support':fi[support_end_col].max(),
                        'threshold_first_raw_support':th[support_start_col].min(),
                        'raw_history_disjoint':bool(fi[support_end_col].max()<th[support_start_col].min())})
        boundaries.append(row)
    f['phase']=roles
    columns=[building_col,time_col,label_col,fault_col,'phase']
    if has_support: columns += [support_start_col,support_end_col]
    assignments=f[columns].copy()
    assignments['nominal_fraction']=fraction;assignments['purged']=purge
    coverage=(assignments.groupby(['phase',fault_col],dropna=False).size()
              .rename('hours').reset_index().rename(columns={fault_col:'fault_type'}))
    alltypes=sorted(f[fault_col].dropna().astype(str).unique())
    coverage=(coverage.set_index(['phase','fault_type'])
              .reindex(pd.MultiIndex.from_product([['fit','threshold','purged'],alltypes],names=['phase','fault_type']),fill_value=0)
              .reset_index())
    coverage['nominal_fraction']=fraction;coverage['purged']=purge
    return CalibrationSplit(roles=='fit',roles=='threshold',roles=='purged',
                            assignments,pd.DataFrame(boundaries),coverage)


def write_calibration_grid(frame: pd.DataFrame, destination, fractions: Iterable[float]=(.5,.6,.7)):
    """Write six frozen chronology conditions; no score or test result is used."""
    from pathlib import Path
    dest=Path(destination);dest.mkdir(parents=True,exist_ok=True)
    assignments=[];bounds=[];coverage=[]
    for fraction in fractions:
        for purge in (False,True):
            s=make_calibration_split(frame,fraction,purge=purge)
            name=f'cut{int(round(100*fraction)):02d}_'+('strict_history_disjoint' if purge else 'causal_unpurged')
            a=s.assignments.copy();a['condition']=name
            b=s.boundaries.copy();b['condition']=name
            c=s.fault_coverage.copy();c['condition']=name
            assignments.append(a);bounds.append(b);coverage.append(c)
    pd.concat(assignments,ignore_index=True).to_csv(dest/'calibration_assignments.csv',index=False)
    pd.concat(bounds,ignore_index=True).to_csv(dest/'calibration_boundaries.csv',index=False)
    pd.concat(coverage,ignore_index=True).to_csv(dest/'calibration_fault_coverage.csv',index=False)
