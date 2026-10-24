"""Mutually exclusive measured-window exclusion accounting, no forecast scoring."""
import numpy as np
import pandas as pd
from common import CACHE,OUT,arrays,protocol,safe_csv


def main():
    p=protocol();raw=pd.read_csv(CACHE/'bdg2_selected_raw_meter.csv',parse_dates=['timestamp'])
    meta=pd.read_csv(CACHE/'bdg2_selected_metadata.csv').set_index('building_id')
    cached={x:arrays(x) for x in ['train','validation','test']};rows=[]
    time=raw.timestamp
    for bid in p['buildings']:
        meter=raw[bid].where(np.isfinite(raw[bid])&(raw[bid]>=0))
        target=meter.notna().to_numpy();anchor=meter.shift(1).notna().to_numpy()
        history=meter.shift(2).rolling(47,min_periods=47).count().eq(47).to_numpy()
        initial=np.arange(len(raw))<48
        reason=np.select([initial,~target,~anchor,~history],
                         ['initial48h_history','missing_target','missing_anchor','incomplete_prior_window'],default='valid')
        for split,period in [('train',time.lt('2016-11-01')),('validation',time.ge('2016-11-01')&time.lt('2017-01-01')),('test',time.ge('2017-01-01'))]:
            z=period.to_numpy();expected=time[z&(reason=='valid')].to_numpy()
            S=cached[split];actual=np.asarray(S['ts'][S['bid']==bid])
            assert np.array_equal(expected,actual),(bid,split)
            r={'building_id':bid,'site_id':bid.split('_')[0],'primary_use':meta.loc[bid,'primaryspaceusage'],
               'floor_area_sqft':float(meta.loc[bid,'sqft']),'split':split,'calendar_hours':int(z.sum()),
               'raw_nonnegative_meter_hours':int(np.sum(z&target)),
               'valid_scored_targets':int(np.sum(z&(reason=='valid')))}
            for why in ['initial48h_history','missing_target','missing_anchor','incomplete_prior_window']:
                r['excluded_'+why]=int(np.sum(z&(reason==why)))
            r['valid_coverage_pct']=100*r['valid_scored_targets']/r['calendar_hours']
            assert r['valid_scored_targets']+sum(r['excluded_'+why] for why in ['initial48h_history','missing_target','missing_anchor','incomplete_prior_window'])==r['calendar_hours']
            rows.append(r)
    df=pd.DataFrame(rows);safe_csv(df,OUT/'table_building_coverage_exclusions.csv')
    countcols=['calendar_hours','raw_nonnegative_meter_hours','valid_scored_targets','excluded_initial48h_history','excluded_missing_target','excluded_missing_anchor','excluded_incomplete_prior_window']
    totals=df.groupby('split')[countcols].sum().reset_index();safe_csv(totals,OUT/'table_coverage_totals.csv')
    print(totals.to_string(index=False))


if __name__=='__main__':main()
