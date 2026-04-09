"""
Prepara il dataset per il trading sul naphtha crack spread (naphtha - brent).
Feature set ispirato al framework di Scaillet et al. per il trading mean-reversion
su crack spread: standardized shocks, deviation from mean, cross-asset dynamics.
"""

import pandas as pd
import numpy as np
import argparse
import os


def compute_crack_features(crack: pd.Series, naphtha: pd.Series, brent: pd.Series, brent_volume: pd.Series ,
                           vol_window: int = 20, mean_window: int = 20) -> pd.DataFrame:
    """
    Calcola le feature per il mean-reversion trading del crack spread,
    seguendo il framework di Scaillet et al.

    Principi di design:
    - Nessuna feature ridondante (ogni variabile aggiunge informazione unica)
    - Le feature core sono shock standardizzato e deviazione dalla media
    - Le feature cross-asset spiegano CHI muove lo spread
    - Il tutto è pensato per un agente che gestisce posizioni a holding-period fisso

    Feature finali (15 feature di mercato):
        1  delta_crack          — innovazione grezza del processo (d_t)
        2  rolling_vol_crack    — sigma_t, volatilità rolling per standardizzare
        3  shock                — shock standardizzato: d_t / sigma_t (segnale core)
        4  shock_lag1           — autocorrelazione negli shock (timing)
        5  shock_lag2           — secondo lag per catturare decadimento AR
        6  deviation_from_mean  — quanto il crack è fuori equilibrio
        7  zscore_crack         — deviazione normalizzata per la volatilità
        8  momentum_crack_5     — tendenza recente (5d) per overshooting detection
        9  momentum_crack_10    — tendenza più lunga (10d) per regime detection
        10 half_life_proxy      — velocità stimata di mean reversion (autocorr lag-1)
        11 correlation_20       — stabilità della relazione naphtha-brent
        12 vol_ratio_naphtha_brent — quale gamba è più volatile
        13 naphtha_contribution — chi sta guidando il crack oggi
        14 beta_naphtha_brent   — sensibilità strutturale naphtha a brent
        15 shock_x_zscore       — interazione: shock grande + crack fuori equilibrio

    + 3 variabili di stato aggiunte dall'environment:
        16 current_position     — posizione corrente (-1/0/1)
        17 normalized_unrealized_pnl — PnL non realizzato
        18 time_in_position     — da quanti step siamo in posizione (normalizzato)

    Totale stato: 18 dimensioni.
    """
    df = pd.DataFrame(index=crack.index)

    # =========================================================================
    # 1. DELTA CRACK — innovazione grezza del processo
    #    È il cambiamento giornaliero dello spread. Input fondamentale per
    #    calcolare lo shock standardizzato. Tenuto separato perché contiene
    #    informazione sulla scala assoluta che lo shock perde.
    # =========================================================================
    delta_crack = crack.diff()
    df['delta_crack'] = delta_crack

    # =========================================================================
    # 2. ROLLING VOLATILITY — sigma_t
    #    Volatilità rolling dei cambiamenti del crack. Serve a:
    #    (a) standardizzare gli shock (feature 3)
    #    (b) dare all'agente il contesto di rischio corrente
    #    (c) calibrare implicitamente le soglie di entrata
    # =========================================================================
    rolling_vol = delta_crack.rolling(vol_window).std()
    df['rolling_vol_crack'] = rolling_vol

    # =========================================================================
    # 3. STANDARDIZED SHOCK — variabile chiave nel framework Scaillet
    #    shock_t = delta_crack_t / sigma_t
    #    Sotto mean reversion: shock grandi → atteso ritorno alla media.
    #    È il segnale principale per decidere se entrare in posizione.
    #    NOTA: abs_shock rimosso perché derivabile dal segno di shock.
    # =========================================================================
    df['shock'] = delta_crack / (rolling_vol + 1e-8)

    # =========================================================================
    # 4-5. LAGGED SHOCKS — autocorrelazione nel processo
    #    Sotto mean reversion, gli shock hanno autocorrelazione negativa
    #    (uno shock grande è seguito da un ritorno). Due lag catturano
    #    la struttura AR(2) tipica dei processi OU discretizzati.
    # =========================================================================
    df['shock_lag1'] = df['shock'].shift(1)
    df['shock_lag2'] = df['shock'].shift(2)

    # =========================================================================
    # 6. DEVIATION FROM MEAN — distanza dal fair value
    #    Quanto il crack è lontano dalla sua media rolling.
    #    È la variabile di stato del processo OU: x_t - mu.
    #    Grandi deviazioni → forte attrazione verso la media.
    #    NOTA: rolling_mean_crack e crack_level rimossi perché ridondanti
    #    (il livello assoluto non è informativo per mean reversion,
    #     conta solo la deviazione).
    # =========================================================================
    rolling_mean = crack.rolling(mean_window).mean()
    rolling_std = crack.rolling(mean_window).std()
    df['deviation_from_mean'] = crack - rolling_mean

    # =========================================================================
    # 7. Z-SCORE — deviazione normalizzata
    #    Come deviation_from_mean ma normalizzata per la volatilità corrente.
    #    Fondamentale perché la stessa deviazione assoluta ha significato
    #    diverso in regimi di alta vs bassa volatilità.
    # =========================================================================
    df['zscore_crack'] = (crack - rolling_mean) / (rolling_std + 1e-8)

    # =========================================================================
    # 8-9. MOMENTUM — tendenza recente del crack
    #    momentum_5: segnale a breve per overshooting detection
    #    momentum_10: segnale più lungo per regime detection
    df['momentum_crack_5'] = crack - crack.shift(5)
    df['momentum_crack_10'] = crack - crack.shift(10)

    # =========================================================================
    # 10. HALF-LIFE PROXY — velocità di mean reversion
    #    Stimata come l'autocorrelazione lag-1 dei cambiamenti del crack.
    #    Se autocorr ~ -0.5 → mean reversion veloce (half-life corta)
    #    Se autocorr ~ 0   → random walk (no mean reversion)
    #    Se autocorr ~ +0.5 → trending (momentum)
    #    Aiuta l'agente a capire se il regime corrente è mean-reverting
    #    e a calibrare l'holding period atteso.
    # =========================================================================
    df['half_life_proxy'] = delta_crack.rolling(vol_window).apply(
        lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 1 else 0.0,
        raw=True
    )

    # =========================================================================
    # 11. CORRELAZIONE ROLLING NAPHTHA-BRENT
    #    Correlazione alta → crack stabile (gambe si muovono insieme)
    #    Correlazione in calo → divergenza, crack più volatile
    #    Segnale di regime: bassa correlazione = più opportunità mean reversion
    # =========================================================================
    naphtha_ret = naphtha.pct_change()
    brent_ret = brent.pct_change()
    df['correlation_20'] = naphtha_ret.rolling(vol_window).corr(brent_ret)

    # =========================================================================
    # 12. RAPPORTO DI VOLATILITÀ NAPHTHA/BRENT
    #    > 1: naphtha più volatile → naphtha guida il crack
    #    < 1: brent più volatile → brent guida il crack
    #    Informazione complementare a naphtha_contribution (feature 13):
    #    questa è strutturale (su 20d), l'altra è puntuale (1d).
    # =========================================================================
    vol_naphtha = naphtha_ret.rolling(vol_window).std()
    vol_brent = brent_ret.rolling(vol_window).std()
    df['vol_ratio_naphtha_brent'] = vol_naphtha / (vol_brent + 1e-8)

    # =========================================================================
    # 13. CONTRIBUTO NAPHTHA AL CAMBIAMENTO DEL CRACK (oggi)
    #    Decompone |d(crack)| in quota naphtha vs quota brent.
    #    ~1 → naphtha domina il movimento di oggi
    #    ~0 → brent domina il movimento di oggi
    #    Informazione puntuale complementare al vol_ratio (strutturale).
    # =========================================================================
    d_naphtha = naphtha.diff()
    d_brent = brent.diff()
    abs_total = d_naphtha.abs() + d_brent.abs() + 1e-8
    df['naphtha_contribution'] = d_naphtha.abs() / abs_total

    # =========================================================================
    # 14. BETA ROLLING NAPHTHA vs BRENT
    #    Beta > 1: naphtha amplifica i movimenti del brent
    #    Beta < 1: naphtha è meno reattiva
    #    Shift nel beta → cambio strutturale nella relazione
    # =========================================================================
    covar = naphtha_ret.rolling(vol_window).cov(brent_ret)
    var_brent = brent_ret.rolling(vol_window).var()
    df['beta_naphtha_brent'] = covar / (var_brent + 1e-8)

    # =========================================================================
    # 15. INTERAZIONE SHOCK × Z-SCORE
    #    Cattura il caso critico per mean reversion: uno shock grande
    #    che avviene quando il crack è già fuori equilibrio.
    #    shock positivo + zscore positivo = overshooting al rialzo → short
    #    shock negativo + zscore negativo = overshooting al ribasso → long
    #    Questa feature non-lineare aiuta la rete a identificare i punti
    #    di ingresso ottimali senza dover "scoprire" l'interazione da sola.
    # =========================================================================
    df['shock_x_zscore'] = df['shock'] * df['zscore_crack']

    df['log_brent_volume'] = np.log(brent_volume + 1.0)
    df['brent_volume_zscore_20'] = (
    (brent_volume - brent_volume.rolling(vol_window).mean()) /
    (brent_volume.rolling(vol_window).std() + 1e-8))

    df['brent_volume_change'] = brent_volume.pct_change()

    return df


def prepare_data(naphtha_path: str = 'Data/naphtha_path.csv', brent_path: str = 'Data/brent_path.csv', output_dir: str = 'Data',
                 date_col: str = 'Date', close_col: str = 'Close',
                 train_ratio: float = 0.8):
    """
    Pipeline principale di preparazione dati.

    Args:
        naphtha_path: CSV con prezzi naphtha
        brent_path: CSV con prezzi brent
        output_dir: cartella output
        date_col: nome colonna data
        close_col: nome colonna prezzo chiusura
        train_ratio: proporzione dati training
    """
    os.makedirs(output_dir, exist_ok=True)

    # Carica dati
    df_naphtha = pd.read_csv(naphtha_path, parse_dates=[date_col])
    df_brent = pd.read_csv(brent_path, parse_dates=[date_col])

    print(f"Naphtha: {len(df_naphtha)} righe, da {df_naphtha[date_col].min()} a {df_naphtha[date_col].max()}")
    print(f"Brent:   {len(df_brent)} righe, da {df_brent[date_col].min()} a {df_brent[date_col].max()}")

    # Rinomina colonne
    df_naphtha = df_naphtha[[date_col, close_col]].rename(columns={close_col: 'Naphtha_Close'})
    df_brent = df_brent[[date_col, close_col, 'PX_VOLUME']].rename(columns={close_col: 'Brent_Close' , 'PX_VOLUME' : 'Brent_Volume'})

    # Merge su data (inner join: solo giorni in comune)
    df = pd.merge(df_naphtha, df_brent, on=date_col, how='inner')
    df = df.sort_values(date_col).reset_index(drop=True)
    print(f"Dopo merge: {len(df)} righe, da {df[date_col].min()} a {df[date_col].max()}")

    # Calcola crack spread
    df['Crack_Spread'] = df['Naphtha_Close'] - df['Brent_Close']

    # Il prezzo di trading è lo spread (Close è usato dall'environment)
    df['Close'] = df['Crack_Spread']

    # Calcola feature con il framework Scaillet-inspired
    print("Calcolo feature crack spread + cross-asset...")
    features = compute_crack_features(
        crack=df['Crack_Spread'],
        naphtha=df['Naphtha_Close'],
        brent=df['Brent_Close'] , 
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
    print(f"Righe rimosse per NaN: {n_before - n_after}")
    print(f"Dataset finale: {n_after} righe, {len(result.columns)} colonne")

    # Split train/test
    split_idx = int(len(result) * train_ratio)
    train_df = result.iloc[:split_idx].reset_index(drop=True)
    test_df = result.iloc[split_idx:].reset_index(drop=True)

    print(f"\nTrain: {len(train_df)} righe ({train_df[date_col].min()} - {train_df[date_col].max()})")
    print(f"Test:  {len(test_df)} righe ({test_df[date_col].min()} - {test_df[date_col].max()})")

    # Stampa colonne feature
    feature_cols = [c for c in result.columns if c not in [date_col, 'Close', 'Naphtha_Close', 'Brent_Close'  , 'Brent_Volume' , 'Crack_Spread']]
    print(f"\nFeature ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols):
        print(f"  {i+1:3d}. {col}")

    # Salva
    train_path = os.path.join(output_dir, 'naphtha_crack_train.csv')
    test_path = os.path.join(output_dir, 'naphtha_crack_test.csv')
    full_path = os.path.join(output_dir, 'naphtha_crack_full.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    result.to_csv(full_path, index=False)

    print(f"\nFile salvati:")
    print(f"  Train: {train_path}")
    print(f"  Test:  {test_path}")
    print(f"  Full:  {full_path}")

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
    
    print(f"📂 Loading naphtha from: {args.naphtha}", flush=True)
    print(f"📂 Loading brent from: {args.brent}", flush=True)
    
    prepare_data(
        naphtha_path= 'Data/naphtha_path.csv',
        brent_path='Data/brent_path.csv',
        output_dir= 'Data',
        date_col=args.date_col,
        close_col=args.close_col,
        train_ratio=args.train_ratio
    )
    
    print(f"Merged data", flush=True)
    print(f"Features computed columns", flush=True)
    print(f"Train set: rows | Test set: rows", flush=True)
    print(f"Saved to {args.output_dir}/", flush=True)
