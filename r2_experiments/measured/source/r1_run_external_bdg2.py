"""Independent BDG2 measured-electricity, chronological one-hour forecast check.

This is a retrained telemetry-input version of the existing DRA/PSNN architecture;
it is NOT zero-shot transfer of the synthetic MOA feature mask, nor validation of
synthetic fault labels or controller savings. See protocol.json and REPORT.md.
"""
import ast
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = HERE / 'raw'
OUT = HERE / 'results'
OUT.mkdir(exist_ok=True)
torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)
L = 24
SEEDS = [11, 23, 37]
FEATURES = ['airTemperature', 'dewTemperature', 'log_area', 'hour_sin',
            'hour_cos', 'dow_sin', 'dow_cos', 'lag1', 'lag2', 'lag6',
            'lag12', 'lag24', 'roll24']
H = {'channels': 20, 'lr': 0.0015287143259818435, 'dropout': .08,
     'beta': 0.9431572529176961, 'theta': 0.36879562783560627,
     'tau': .18, 'kappa': .35}


def exact_architecture_classes():
    """Execute only model class AST nodes; never execute original experiments."""
    src = (ROOT / 'scripts/run_canonical_s105.py').read_text()
    names = ['RAB', 'HDRAB', 'DRA', 'PSNN', 'Proposed', 'DRAGRU',
             'DRALSTM', 'Transformer']
    nodes = [node for node in ast.parse(src).body
             if isinstance(node, ast.ClassDef) and node.name in names]
    assert len(nodes) == len(names)
    extracted = '\n\n'.join(ast.get_source_segment(src, node) for node in nodes)
    (HERE / 'exact_architecture.py').write_text(
        '# Exact class definitions extracted from scripts/run_canonical_s105.py\n'
        'import torch\nimport torch.nn as nn\nL=24\n\n' + extracted + '\n')
    scope = {'torch': torch, 'nn': nn, 'L': L}
    exec(compile(ast.Module(body=nodes, type_ignores=[]),
                 'exact_canonical_architecture', 'exec'), scope)
    return scope, hashlib.sha256(src.encode()).hexdigest()


def prepare():
    metadata = pd.read_csv(RAW / 'bdg2_metadata.csv')
    electricity = pd.read_csv(RAW / 'bdg2_electricity.csv', parse_dates=['timestamp'])
    electricity = electricity.set_index('timestamp').sort_index()
    assert electricity.index.is_unique
    expected = pd.date_range('2016-01-01', '2017-12-31 23:00:00', freq='h')
    assert electricity.index.equals(expected)
    training = electricity.loc[:'2016-10-31 23:00:00']

    # Decide eligibility and subset using only training-year observations.
    eligible = []
    for _, row in metadata.sort_values('building_id').iterrows():
        bid = row.building_id
        if bid not in training or not np.isfinite(row.sqft) or row.sqft <= 0:
            continue
        x = training[bid].to_numpy(float)
        valid = np.isfinite(x) & (x >= 0)
        if valid.mean() >= .95 and np.nanmedian(x) > 1 and np.nanstd(x) > 1:
            eligible.append(bid)
    sites = []
    for site, group in metadata[metadata.building_id.isin(eligible)].groupby('site_id', sort=True):
        if len(group) >= 2:
            sites.append(site)
    sites = sites[:6]
    selected = []
    for site in sites:
        candidates = metadata[metadata.site_id.eq(site) & metadata.building_id.isin(eligible)].sort_values('building_id')
        chosen = []
        for use in ['Education', 'Office']:
            ids = candidates.loc[candidates.primaryspaceusage.eq(use), 'building_id'].tolist()
            if ids:
                chosen.append(ids[0])
        for bid in candidates.building_id:
            if len(chosen) >= 2:
                break
            if bid not in chosen:
                chosen.append(bid)
        selected.extend(chosen)
    assert len(selected) == 12 and len(set(selected)) == 12
    metadata = metadata[metadata.building_id.isin(selected)].copy()
    weather = pd.read_csv(RAW / 'bdg2_weather.csv', parse_dates=['timestamp'])
    weather = weather[weather.site_id.isin(sites)].copy()
    assert not weather.duplicated(['site_id', 'timestamp']).any()
    electricity[selected].reset_index().to_csv(HERE / 'bdg2_selected_raw_meter.csv', index=False)
    metadata.to_csv(HERE / 'bdg2_selected_metadata.csv', index=False)
    weather.to_csv(HERE / 'bdg2_selected_weather.csv', index=False)
    frames = []
    coverage = []
    for bid in selected:
        row = metadata.loc[metadata.building_id.eq(bid)].iloc[0]
        g = pd.DataFrame({'timestamp': expected, 'meter': electricity[bid].to_numpy()})
        g['building_id'] = bid
        g['site_id'] = row.site_id
        g['square_feet'] = row.sqft
        g['meter'] = g.meter.where(np.isfinite(g.meter) & (g.meter >= 0))
        w = weather.loc[weather.site_id.eq(row.site_id)].set_index('timestamp').reindex(expected)
        for f in ['airTemperature', 'dewTemperature']:
            s = w[f].astype(float)
            train_median = s.loc[:'2016-10-31 23:00:00'].median()
            assert np.isfinite(train_median), (row.site_id, f)
            # Past-only fill, at most 6 hours; remaining gaps use train-only median.
            g[f] = s.ffill(limit=6).fillna(train_median).to_numpy()
        g['log_area'] = np.log1p(g.square_feet)
        g['hour_sin'] = np.sin(2*np.pi*g.timestamp.dt.hour/24)
        g['hour_cos'] = np.cos(2*np.pi*g.timestamp.dt.hour/24)
        g['dow_sin'] = np.sin(2*np.pi*g.timestamp.dt.dayofweek/7)
        g['dow_cos'] = np.cos(2*np.pi*g.timestamp.dt.dayofweek/7)
        for lag in [1, 2, 6, 12, 24]:
            g[f'lag{lag}'] = g.meter.shift(lag)
        g['roll24'] = g.meter.shift(1).rolling(24, min_periods=24).mean()
        frames.append(g)
        for year in [2016, 2017]:
            z = g[g.timestamp.dt.year.eq(year)]
            coverage.append([bid, row.site_id, row.primaryspaceusage, row.sqft,
                             year, len(z), int(z.meter.notna().sum()),
                             int(z.meter.eq(0).sum()), z.meter.mean()])
    frame = pd.concat(frames, ignore_index=True)
    pd.DataFrame(coverage, columns=['building_id', 'site_id', 'primary_use',
                                  'area_sqft', 'year', 'hours', 'valid_raw_meter',
                                  'zero_meter_hours', 'mean_meter_kWh']).to_csv(OUT / 'data_coverage.csv', index=False)
    train_rows = frame[frame.timestamp.lt('2016-11-01')].dropna(subset=FEATURES)
    scaler = StandardScaler().fit(train_rows[FEATURES])
    records = {'train': [], 'validation': [], 'test': []}
    for bid, g in frame.groupby('building_id', sort=False):
        g = g.sort_values('timestamp').reset_index(drop=True)
        a = scaler.transform(g[FEATURES]).astype('float32')
        # At target t, the model receives feature rows [t-24,t-1] only;
        # load reconstruction separately uses observed meter(t-1).
        for i in range(L, len(g)):
            X = a[i-L:i]
            anchor = g.meter.iloc[i-1]
            target = g.meter.iloc[i]
            if not np.isfinite(X).all() or not np.isfinite(anchor) or not np.isfinite(target):
                continue
            ts = g.timestamp.iloc[i]
            split = 'train' if ts < pd.Timestamp('2016-11-01') else 'validation' if ts < pd.Timestamp('2017-01-01') else 'test'
            records[split].append((X, float(target), float(anchor),
                                   float(g.square_feet.iloc[i]), bid, g.site_id.iloc[i], ts))
    result = {}
    splitrows = []
    for split, items in records.items():
        result[split] = {'X': np.stack([x[0] for x in items]),
                         'target': np.array([x[1] for x in items]),
                         'anchor': np.array([x[2] for x in items]),
                         'area': np.array([x[3] for x in items]),
                         'building_id': np.array([x[4] for x in items]),
                         'site_id': np.array([x[5] for x in items]),
                         'timestamp': pd.to_datetime([x[6] for x in items])}
        S = result[split]
        splitrows.append([split, str(S['timestamp'].min()), str(S['timestamp'].max()), len(items)])
    trdelta = (result['train']['target']-result['train']['anchor'])/np.sqrt(result['train']['area'])
    mu, sd = float(trdelta.mean()), float(trdelta.std(ddof=1)+1e-9)
    for S in result.values():
        S['y'] = (((S['target']-S['anchor'])/np.sqrt(S['area'])-mu)/sd).astype('float32')
    pd.DataFrame(splitrows, columns=['split', 'first_target', 'last_target', 'sequence_rows']).to_csv(OUT / 'split_protocol.csv', index=False)
    assert result['train']['timestamp'].max() < result['validation']['timestamp'].min()
    assert result['validation']['timestamp'].max() < result['test']['timestamp'].min()
    np.savez(HERE / 'normalization.npz', mean=scaler.mean_, scale=scaler.scale_, delta_mean=mu, delta_sd=sd)
    return result, mu, sd, selected, sites


def train(model, X, y, seed):
    # Original canonical train() resets the RNG after model initialization.
    # Keep identical minibatch ordering across architectures for each seed.
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=H['lr'], weight_decay=1e-4)
    xt, yt = torch.tensor(X), torch.tensor(y)
    model.train()
    for _ in range(5):
        perm = torch.randperm(len(xt))
        for j in range(0, len(xt), 256):
            z = perm[j:j+256]
            loss = ((model(xt[z])-yt[z])**2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
    return model


def predict(model, S, mu, sd):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(S['X']), 1024):
            out.append(model(torch.tensor(S['X'][i:i+1024])).numpy())
    return S['anchor']+(np.concatenate(out)*sd+mu)*np.sqrt(S['area'])


def metrics(y, p):
    rmse = float(np.sqrt(mean_squared_error(y, p)))
    return [rmse, float(mean_absolute_error(y, p)), float(r2_score(y, p)),
            float(100*rmse/np.mean(y))]


def save_compressed(frame, path):
    # Fully close and verify the compressed stream before publishing its name.
    import gzip
    temp = path.with_suffix('.partial.gz')
    frame.to_csv(temp, index=False)
    with gzip.open(temp, 'rb') as f:
        while f.read(1024*1024):
            pass
    temp.replace(path)


def run():
    models, sourcehash = exact_architecture_classes()
    parts, mu, sd, selected, sites = prepare()
    protocol = {
        'dataset': 'Building Data Genome 2 raw measured electricity',
        'dataset_doi': '10.1038/s41597-020-00712-x',
        'repository_commit': '9b97ccbe90096aff42ed4fd6493bf7ae692d7118',
        'selection': 'First six lexicographic sites with >=2 eligible buildings; first Education and Office building per site where available, lexicographic fallback; eligibility uses ONLY Jan-Oct 2016: >=95% nonnegative finite meter values, median >1 kWh, SD >1 kWh, positive published floor area.',
        'selected_buildings': selected, 'sites': sites,
        'train': '2016-01-01 to 2016-10-31',
        'validation': '2016-11-01 to 2016-12-31',
        'test': '2017-01-01 to 2017-12-31, full 365 calendar days',
        'window_hours': L, 'horizon_hours': 1, 'features': FEATURES,
        'missingness': 'No target/meter imputation; drop sequences with missing target, anchor, or lag/window inputs. Weather causal ffill <=6h, then training-only site median; no backward fill.',
        'model': 'Exact original DRA and PSNN class implementations, 13 available-input version; original canonical hyperparameters frozen, no MOA/DOA rerun, no test-tuned choice.',
        'canonical_source_sha256': sourcehash, 'hyperparameters': H,
        'comparison': 'Encoder-matched DRA-GRU and DRA-LSTM (ch=h=20, dropout .08), Transformer (d=24, dropout .08), Persistence; identical train samples and five epochs, AdamW/clip settings and 3 seeds.',
        'training_sequence_budget': 6800, 'sampling': 'deterministic evenly spaced training sequences',
        'checkpoint_selection': 'Final epoch, fixed a priori; validation used for reported diagnostics only, no model/test selection.',
        'seeds': SEEDS, 'python': platform.python_version(),
        'torch': torch.__version__, 'numpy': np.__version__,
        'pandas': pd.__version__, 'threads': torch.get_num_threads(),
        'scope': 'Dataset-external retraining and later-year forecasting, not zero-shot transfer, complete BDG2 benchmarking, physical IoT/BAS validation, fault-label validation, or controller validation.'}
    (HERE / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    ii = np.linspace(0, len(parts['train']['X'])-1, min(6800, len(parts['train']['X'])), dtype=int)
    np.save(OUT / 'training_sequence_indices.npy', ii)
    rows, bybuilding, byseason, timings, daily = [], [], [], [], []
    te = parts['test']
    meta = pd.DataFrame({'building_id': te['building_id'], 'site_id': te['site_id'],
                         'timestamp': te['timestamp'], 'actual_kWh': te['target'],
                         'persistence_kWh': te['anchor']})
    save_compressed(meta, OUT / 'test_rows.csv.gz')
    for name in ['DRA-PSNN (measured telemetry)', 'DRA-GRU', 'DRA-LSTM', 'Transformer']:
        for seed in SEEDS:
            start = time.perf_counter()
            torch.manual_seed(seed)
            np.random.seed(seed)
            if name.startswith('DRA-PSNN'):
                model = models['Proposed'](len(FEATURES), H['channels'], H['dropout'], H['beta'], H['theta'], H['tau'], H['kappa'])
            elif name == 'DRA-GRU':
                model = models['DRAGRU'](len(FEATURES), ch=20, drop=.08, h=20)
            elif name == 'DRA-LSTM':
                model = models['DRALSTM'](len(FEATURES), ch=20, drop=.08, h=20)
            else:
                model = models['Transformer'](len(FEATURES), d=24, drop=.08)
            slug = 'DRA_PSNN' if name.startswith('DRA-PSNN') else name.replace('-', '_')
            train(model, parts['train']['X'][ii], parts['train']['y'][ii], seed)
            train_seconds = time.perf_counter()-start
            torch.save(model.state_dict(), OUT / f'{slug}_seed{seed}.pt')
            for split in ['validation', 'test']:
                S = parts[split]
                pp = predict(model, S, mu, sd)
                rows.append([name, seed, split, len(pp), *metrics(S['target'], pp)])
                prediction = pd.DataFrame({'building_id': S['building_id'], 'timestamp': S['timestamp'], 'actual_kWh': S['target'], 'prediction_kWh': pp})
                save_compressed(prediction, OUT / f'{slug}_seed{seed}_{split}_predictions.csv.gz')
                if split != 'test':
                    continue
                for bid in selected:
                    z = S['building_id'] == bid
                    if z.any():
                        bybuilding.append([name, seed, bid, len(pp[z]), *metrics(S['target'][z], pp[z])])
                for season, months in [('DJF', [12,1,2]), ('MAM', [3,4,5]), ('JJA', [6,7,8]), ('SON', [9,10,11])]:
                    z = np.isin(S['timestamp'].month, months)
                    byseason.append([name, seed, season, int(z.sum()), *metrics(S['target'][z], pp[z])])
                dd = prediction.assign(day=prediction.timestamp.dt.floor('D'))
                for (bid, day), g in dd.groupby(['building_id', 'day']):
                    if len(g) >= 18:
                        daily.append([name, seed, bid, day, len(g), float(np.mean(np.abs(g.actual_kWh-g.prediction_kWh)))])
            seconds = time.perf_counter()-start
            timings.append([name, seed, train_seconds, seconds, sum(p.numel() for p in model.parameters())])
            print(name, seed, 'train_s', round(train_seconds,2), 'total_s', round(seconds,2), rows[-1], flush=True)
            pd.DataFrame(rows, columns=['model','seed','split','n','RMSE_kWh','MAE_kWh','R2','CVRMSE_pct']).to_csv(OUT / 'forecast_metrics_seed.csv', index=False)
    for split in ['validation', 'test']:
        S = parts[split]
        rows.append(['Persistence', 0, split, len(S['target']), *metrics(S['target'], S['anchor'])])
    for bid in selected:
        z = te['building_id'] == bid
        if z.any():
            bybuilding.append(['Persistence',0,bid,int(z.sum()),*metrics(te['target'][z],te['anchor'][z])])
    for season, months in [('DJF',[12,1,2]),('MAM',[3,4,5]),('JJA',[6,7,8]),('SON',[9,10,11])]:
        z = np.isin(te['timestamp'].month, months)
        byseason.append(['Persistence',0,season,int(z.sum()),*metrics(te['target'][z],te['anchor'][z])])
    columns = ['model','seed','split','n','RMSE_kWh','MAE_kWh','R2','CVRMSE_pct']
    report = pd.DataFrame(rows, columns=columns)
    report.to_csv(OUT / 'forecast_metrics_seed.csv', index=False)
    report.groupby(['model','split'])[['RMSE_kWh','MAE_kWh','R2','CVRMSE_pct']].agg(['mean','std']).to_csv(OUT / 'forecast_metrics_summary.csv')
    pd.DataFrame(bybuilding,columns=['model','seed','building_id','n','RMSE_kWh','MAE_kWh','R2','CVRMSE_pct']).to_csv(OUT / 'forecast_metrics_building.csv',index=False)
    pd.DataFrame(byseason,columns=['model','seed','season','n','RMSE_kWh','MAE_kWh','R2','CVRMSE_pct']).to_csv(OUT / 'forecast_metrics_season.csv',index=False)
    pd.DataFrame(timings,columns=['model','seed','train_seconds','total_seconds','parameters']).to_csv(OUT / 'timings.csv',index=False)
    save_compressed(pd.DataFrame(daily,columns=['model','seed','building_id','day','n','MAE_kWh']),OUT / 'daily_MAE.csv.gz')
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    run()
