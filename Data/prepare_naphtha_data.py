"""
Prepara il dataset per il trading sul naphtha crack spread (naphtha - brent).
Feature set ispirato al framework di Scaillet et al. per il trading mean-reversion
su crack spread: standardized shocks, deviation from mean, cross-asset dynamics.
"""

import pandas as pd
import numpy as np
import argparse
import os


def compute_crack_features(crack:pd.Series,
                            naphtha: pd.Series,
                            brent: pd.Series, 
                            brent_volume: pd.Series ,
                            vol_window: int = 20,
                             mean_window: int = 20) -> pd.DataFrame:
    """
    Calcola le feature per il mean-reversion trading del crack spread,
    seguendo il framework di Scaillet et al.

    Principi :
    - Le feature core sono shock standardizzato e deviazione dalla media
    - Le feature cross-asset spiegano CHI muove lo spread
    - Il tutto è pensato per un agente che gestisce posizioni a holding-period fisso

    """
    df = pd.DataFrame(index=crack.index)

    
    #  DELTA CRACK — innovazione grezza del processo
    #    È il cambiamento giornaliero dello spread. Input fondamentale per
    #    calcolare lo shock standardizzato. Tenuto separato perché contiene
    #    informazione sulla scala assoluta che lo shock perde.
    
    delta_crack = crack.diff()
    df['delta_crack'] = delta_crack

    
    #  ROLLING VOLATILITY — sigma_t
    #    Volatilità rolling dei cambiamenti del crack. Serve a:
    #    (a) standardizzare gli shock (feature 3)
    #    (b) dare all'agente il contesto di rischio corrente
    #    (c) calibrare implicitamente le soglie di entrata
    
    rolling_vol = delta_crack.rolling(vol_window).std()
    df['rolling_vol_crack'] = rolling_vol

    
    # STANDARDIZED SHOCK — variabile chiave nel framework Scaillet
    #    shock_t = delta_crack_t / sigma_t
    #    Sotto mean reversion: shock grandi → atteso ritorno alla media.
    #    È il segnale principale per decidere se entrare in posizione.
    #    NOTA: abs_shock rimosso perché derivabile dal segno di shock.
    
    df['shock'] = delta_crack / (rolling_vol + 1e-8)

    
    #  LAGGED SHOCKS — autocorrelazione nel processo
    #    Sotto mean reversion, gli shock hanno autocorrelazione negativa
    #    (uno shock grande è seguito da un ritorno). Due lag catturano
    #    la struttura AR(2) tipica dei processi OU discretizzati.
    
    df['shock_lag1'] = df['shock'].shift(1)

    
    # DEVIATION FROM MEAN — distanza dal fair value
    #    Quanto il crack è lontano dalla sua media rolling.
    #    È la variabile di stato del processo OU: x_t - mu.
    #    Grandi deviazioni → forte attrazione verso la media.
    #    NOTA: rolling_mean_crack e crack_level rimossi perché ridondanti
    #    (il livello assoluto non è informativo per mean reversion,
    #     conta solo la deviazione).
    
    rolling_mean = crack.rolling(mean_window).mean()
    rolling_std = crack.rolling(mean_window).std()
    df['deviation_from_mean'] = crack - rolling_mean

    
    # Z-SCORE — deviazione normalizzata
    #    Come deviation_from_mean ma normalizzata per la volatilità corrente.
    #    Fondamentale perché la stessa deviazione assoluta ha significato
    #    diverso in regimi di alta vs bassa volatilità.
    
    df['zscore_crack'] = (crack - rolling_mean) / (rolling_std + 1e-8)

    

    
    # HALF-LIFE PROXY — velocità di mean reversion
    #    Stimata come l'autocorrelazione lag-1 dei cambiamenti del crack.
    #    Se autocorr ~ -0.5 → mean reversion veloce (half-life corta)
    #    Se autocorr ~ 0   → random walk (no mean reversion)
    #    Se autocorr ~ +0.5 → trending (momentum)
    #    Aiuta l'agente a capire se il regime corrente è mean-reverting
    #    e a calibrare l'holding period atteso.
    
    df['half_life_proxy'] = delta_crack.rolling(vol_window).apply(
        lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 1 else 0.0,
        raw=True
    )





   

    
    

    df['log_brent_volume'] = np.log(brent_volume + 1.0)
    df['brent_volume_zscore_20'] = (
    (brent_volume - brent_volume.rolling(vol_window).mean()) /
    (brent_volume.rolling(vol_window).std() + 1e-8))

    df['brent_volume_change'] = brent_volume.pct_change()

    df['shock_x_volume'] = abs(df['shock']) * df['brent_volume_zscore_20']

   

    return df


def prepare_data(naphtha_path: str = 'Data/naphtha_path.csv', brent_path: str = 'Data/brent_path.csv', output_dir: str = 'Data',
                 date_col: str = 'Date', close_col: str = 'Close',
                 train_ratio: float = 0.8):
    
    os.makedirs(output_dir, exist_ok=True)

    # Carica dati
    df_naphtha = pd.read_csv(naphtha_path, parse_dates=[date_col])
    df_brent = pd.read_csv(brent_path, parse_dates=[date_col])

    # Rinomina colonne
    df_naphtha = df_naphtha[[date_col, close_col]].rename(columns={close_col: 'Naphtha_Close'})
    df_brent = df_brent[[date_col, close_col, 'PX_VOLUME']].rename(columns={close_col: 'Brent_Close' , 'PX_VOLUME' : 'Brent_Volume'})

    # Merge su data (inner join: solo giorni in comune)
    df = pd.merge(df_naphtha, df_brent, on=date_col, how='inner')
    df = df.sort_values(date_col).reset_index(drop=True)

    # Calcola crack spread
    df['Crack_Spread'] = df['Naphtha_Close'] - df['Brent_Close']

    # Il prezzo di trading è lo spread (Close è usato dall'environment)
    df['Close'] = df['Crack_Spread']

        # Calcola feature
    features = compute_crack_features(
        crack=df['Crack_Spread'],
        naphtha=df['Naphtha_Close'],
        brent=df['Brent_Close'],
        brent_volume=df['Brent_Volume']
    )


    # Assembla dataset finale
    result = pd.concat([
        df[[date_col, 'Close', 'Naphtha_Close', 'Brent_Close', 'Brent_Volume', 'Crack_Spread']],
        features
    ], axis=1)

    # Rimuovi righe con NaN (dovute a finestre rolling)
    n_before = len(result)
    result = result.dropna().reset_index(drop=True)
    n_after = len(result)

    # Split train/test
    split_idx = int(len(result) * train_ratio)
    train_df = result.iloc[:split_idx].reset_index(drop=True)
    test_df = result.iloc[split_idx:].reset_index(drop=True)

    # Stampa colonne feature
    feature_cols = [c for c in result.columns if c not in [date_col, 'Close', 'Naphtha_Close', 'Brent_Close'  , 'Brent_Volume' , 'Crack_Spread']]
    

    # Salva
    train_path = os.path.join(output_dir, 'naphtha_crack_train.csv')
    test_path = os.path.join(output_dir, 'naphtha_crack_test.csv')
    full_path = os.path.join(output_dir, 'naphtha_crack_full.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    result.to_csv(full_path, index=False)


    return train_df, test_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preparazione dati naphtha crack spread')
    parser.add_argument('--naphtha', type=str, help='CSV prezzi naphtha (Date, Close)')
    parser.add_argument('--brent', type=str, help='CSV prezzi brent (Date, Close)')
    parser.add_argument('--output_dir', type=str, default='data', help='Cartella output')
    parser.add_argument('--date_col', type=str, default='Date', help='Nome colonna data')
    parser.add_argument('--close_col', type=str, default='Close', help='Nome colonna prezzo chiusura')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Proporzione dati training')
    args = parser.parse_args()
    
    
    prepare_data(
        naphtha_path= 'Data/naphtha_path.csv',
        brent_path='Data/brent_path.csv',
        output_dir= 'Data',
        date_col=args.date_col,
        close_col=args.close_col,
        train_ratio=args.train_ratio
    )
    
    
