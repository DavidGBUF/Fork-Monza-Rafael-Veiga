import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

RESULTS_DIR = '../results/'

plt.rcParams.update({
    'figure.figsize': (12, 5),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 11,
})

# ===========================================================================
# 1. Listar arquivos de resultados disponíveis
# ===========================================================================

h5_files = sorted(glob.glob(os.path.join(RESULTS_DIR, '*.h5')))
csv_metrics = sorted(glob.glob(os.path.join(RESULTS_DIR, 'metrics_*.csv')))
csv_defense = sorted(glob.glob(os.path.join(RESULTS_DIR, 'defense_*.csv')))
csv_norms = sorted(glob.glob(os.path.join(RESULTS_DIR, 'norms_*.csv')))

print(f'Arquivos H5 ({len(h5_files)}):')
for f in h5_files:
    print(f'  {os.path.basename(f)}')

print(f'\nCSVs de métricas ({len(csv_metrics)}):')
for f in csv_metrics:
    print(f'  {os.path.basename(f)}')

print(f'\nCSVs de defesa ({len(csv_defense)}):')
for f in csv_defense:
    print(f'  {os.path.basename(f)}')

print(f'\nCSVs de normas ({len(csv_norms)}):')
for f in csv_norms:
    print(f'  {os.path.basename(f)}')

# ===========================================================================
# 2. Carregar e inspecionar um arquivo H5
# ===========================================================================

H5_INDEX = -1

if h5_files:
    h5_path = h5_files[H5_INDEX]
    print(f'\nArquivo: {os.path.basename(h5_path)}\n')

    with h5py.File(h5_path, 'r') as hf:
        print('--- Datasets ---')
        for key in hf.keys():
            data = np.array(hf[key])
            print(f'  {key}: shape={data.shape}, min={data.min():.4f}, max={data.max():.4f}')

        print('\n--- Configuração do Experimento ---')
        for attr_name, attr_val in hf.attrs.items():
            print(f'  {attr_name}: {attr_val}')
else:
    print('Nenhum arquivo H5 encontrado.')

# ===========================================================================
# 3. Plot: Accuracy, AUC e Loss por rodada (do H5)
# ===========================================================================

if h5_files:
    with h5py.File(h5_files[H5_INDEX], 'r') as hf:
        test_acc = np.array(hf['rs_test_acc']) if 'rs_test_acc' in hf else np.array([])
        test_auc = np.array(hf['rs_test_auc']) if 'rs_test_auc' in hf else np.array([])
        train_loss = np.array(hf['rs_train_loss']) if 'rs_train_loss' in hf else np.array([])
        algo = hf.attrs.get('algorithm', '?')
        cc = hf.attrs.get('cluster_comparation', '?')
        nmal = hf.attrs.get('n_client_malicious', '?')

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    title = f'{algo} | cc={cc} | nmal={nmal}'

    if len(test_acc) > 0:
        axes[0].plot(test_acc, color='#2196F3', linewidth=1.5)
        axes[0].set_title('Test Accuracy')
        axes[0].set_xlabel('Evaluation Round')
        axes[0].set_ylabel('Accuracy')
        best = test_acc.max()
        axes[0].axhline(y=best, color='red', linestyle='--', alpha=0.5, label=f'Best: {best:.4f}')
        axes[0].legend()

    if len(test_auc) > 0:
        axes[1].plot(test_auc, color='#4CAF50', linewidth=1.5)
        axes[1].set_title('Test AUC')
        axes[1].set_xlabel('Evaluation Round')
        axes[1].set_ylabel('AUC')
        best_auc = test_auc.max()
        axes[1].axhline(y=best_auc, color='red', linestyle='--', alpha=0.5, label=f'Best: {best_auc:.4f}')
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, 'AUC não disponível', ha='center', va='center', transform=axes[1].transAxes)

    if len(train_loss) > 0:
        axes[2].plot(train_loss, color='#FF5722', linewidth=1.5)
        axes[2].set_title('Train Loss')
        axes[2].set_xlabel('Evaluation Round')
        axes[2].set_ylabel('Loss')
        min_loss = train_loss.min()
        axes[2].axhline(y=min_loss, color='blue', linestyle='--', alpha=0.5, label=f'Min: {min_loss:.4f}')
        axes[2].legend()

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'plots', 'acc_auc_loss.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nPlot salvo: plots/acc_auc_loss.png')
else:
    print('Nenhum arquivo H5 encontrado.')

# ===========================================================================
# 4. Plot: Métricas por rodada (do CSV)
# ===========================================================================

if csv_metrics:
    df_metrics = pd.read_csv(csv_metrics[-1])
    print(f'\nArquivo: {os.path.basename(csv_metrics[-1])}')
    print(df_metrics.head(10))
    print(f'Total de linhas: {len(df_metrics)}')

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(df_metrics['round'], df_metrics['test_acc'], color='#2196F3', marker='.', markersize=3, linewidth=1)
    axes[0].set_title('Test Accuracy')
    axes[0].set_xlabel('Round')

    axes[1].plot(df_metrics['round'], df_metrics['test_auc'], color='#4CAF50', marker='.', markersize=3, linewidth=1)
    axes[1].set_title('Test AUC')
    axes[1].set_xlabel('Round')

    axes[2].plot(df_metrics['round'], df_metrics['train_loss'], color='#FF5722', marker='.', markersize=3, linewidth=1)
    axes[2].set_title('Train Loss')
    axes[2].set_xlabel('Round')

    plt.suptitle('Métricas por Rodada (CSV)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'plots', 'metrics_csv.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plot salvo: plots/metrics_csv.png')
else:
    print('Nenhum CSV de métricas encontrado.')

# ===========================================================================
# 5. Plot: FPR e FRR da defesa (do CSV)
# ===========================================================================

if csv_defense:
    df_defense = pd.read_csv(csv_defense[-1])
    print(f'\nArquivo: {os.path.basename(csv_defense[-1])}')
    print(df_defense.head(10))
    print(f'Total de linhas: {len(df_defense)}')

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(df_defense['round'], df_defense['FPR'], color='#E91E63', linewidth=1.5, label='FPR')
    axes[0].plot(df_defense['round'], df_defense['FRR'], color='#9C27B0', linewidth=1.5, label='FRR')
    axes[0].set_title('FPR e FRR por Rodada')
    axes[0].set_xlabel('Round')
    axes[0].legend()

    axes[1].plot(df_defense['round'], df_defense['num_removed'], color='#FF9800', linewidth=1.5)
    axes[1].set_title('Clientes Removidos por Rodada')
    axes[1].set_xlabel('Round')
    axes[1].set_ylabel('N Removidos')

    window = min(10, len(df_defense))
    if window > 1:
        axes[2].plot(df_defense['round'], df_defense['FPR'].rolling(window).mean(), color='#E91E63', linewidth=1.5, label=f'FPR (média {window}r)')
        axes[2].plot(df_defense['round'], df_defense['FRR'].rolling(window).mean(), color='#9C27B0', linewidth=1.5, label=f'FRR (média {window}r)')
        axes[2].set_title(f'FPR/FRR Média Móvel ({window} rounds)')
        axes[2].set_xlabel('Round')
        axes[2].legend()
    else:
        axes[2].text(0.5, 0.5, 'Poucos dados para média móvel', ha='center', va='center', transform=axes[2].transAxes)

    plt.suptitle('Defesa - FPR/FRR', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'plots', 'defense_fpr_frr.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plot salvo: plots/defense_fpr_frr.png')
else:
    print('Nenhum CSV de defesa encontrado.')

# ===========================================================================
# 5.5. Plot: Normas L2, L3 e Cosseno (do norms CSV)
# ===========================================================================

csv_norms = sorted(glob.glob(os.path.join(RESULTS_DIR, 'norms_*.csv')))

print(f'\nArquivos norms ({len(csv_norms)}):')
for f in csv_norms:
    print(f'  {os.path.basename(f)}')

if csv_norms:
    NORMS_INDEX = -1
    df_norms = pd.read_csv(csv_norms[NORMS_INDEX])
    print(f'\nArquivo: {os.path.basename(csv_norms[NORMS_INDEX])}')
    print(df_norms.head(10))
    print(f'Total de linhas: {len(df_norms)}')
    print(f'Clientes: {df_norms["client_id"].nunique()}')
    print(f'Rounds: {df_norms["round"].nunique()}')

if csv_norms and len(df_norms) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    base_name = os.path.basename(csv_norms[NORMS_INDEX]).replace('norms_', '').replace('.csv', '')
    fig.suptitle(f'Normas - {base_name}', fontsize=14, fontweight='bold')

    ax_cos, ax_l2 = axes[0, 0], axes[0, 1]
    ax_l3, ax_scatter = axes[1, 0], axes[1, 1]

    for cid in df_norms['client_id'].unique():
        cdf = df_norms[df_norms['client_id'] == cid]
        cdf = cdf.sort_values('round')
        mal = cdf['is_malicious'].iloc[0]
        style = '--' if mal else '-'
        lbl = f'C{cid}{"*" if mal else ""}'
        alpha = 0.8 if mal else 0.5
        ax_cos.plot(cdf['round'], cdf['cosine_score'], style, label=lbl, alpha=alpha)
        ax_l2.plot(cdf['round'], cdf['l2_norm'], style, label=lbl, alpha=alpha)
        ax_l3.plot(cdf['round'], cdf['l3_norm'], style, label=lbl, alpha=alpha)

    ax_cos.set_title('Cosine Similarity per Client')
    ax_cos.set_xlabel('Round')
    ax_cos.legend(fontsize=6, ncol=4)

    ax_l2.set_title('L2 Norm per Client')
    ax_l2.set_xlabel('Round')
    ax_l2.legend(fontsize=6, ncol=4)

    ax_l3.set_title('L3 Norm per Client')
    ax_l3.set_xlabel('Round')
    ax_l3.legend(fontsize=6, ncol=4)

    last_round = df_norms['round'].max()
    last = df_norms[df_norms['round'] == last_round]
    for _, r in last.iterrows():
        m = 'x' if r['is_malicious'] else 'o'
        s = 80 if r['is_malicious'] else 40
        ax_scatter.scatter(r['cosine_score'], r['l3_norm'], marker=m, s=s, label=f"C{int(r['client_id'])}{'*' if r['is_malicious'] else ''}")
    ax_scatter.axvline(last['threshold_cos'].iloc[0], color='gray', linestyle='--', alpha=0.5, label='Threshold Cos')
    ax_scatter.axhline(last['threshold_l3'].iloc[0], color='orange', linestyle='--', alpha=0.5, label='Threshold L3')
    ax_scatter.set_title(f'Cosine vs L3 (Round {int(last_round)})')
    ax_scatter.set_xlabel('Cosine Similarity')
    ax_scatter.set_ylabel('L3 Norm')
    ax_scatter.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'plots', 'norms.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plot salvo: plots/norms.png')
else:
    print('Nenhum dado de normas disponível.')

# ===========================================================================
# 6. Comparação entre experimentos
# ===========================================================================

COMPARE_INDICES = list(range(len(h5_files)))

if len(h5_files) >= 2:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx in COMPARE_INDICES:
        if idx >= len(h5_files):
            continue
        with h5py.File(h5_files[idx], 'r') as hf:
            acc = np.array(hf['rs_test_acc']) if 'rs_test_acc' in hf else np.array([])
            loss = np.array(hf['rs_train_loss']) if 'rs_train_loss' in hf else np.array([])
            algo = hf.attrs.get('algorithm', '?')
            cc = hf.attrs.get('cluster_comparation', '?')
            nmal = hf.attrs.get('n_client_malicious', '?')
            label = f'{algo}_cc{cc}_nmal{nmal}'

        if len(acc) > 0:
            axes[0].plot(acc, linewidth=1.5, label=label)
        if len(loss) > 0:
            axes[1].plot(loss, linewidth=1.5, label=label)

    axes[0].set_title('Test Accuracy')
    axes[0].set_xlabel('Evaluation Round')
    axes[0].legend(fontsize=9)

    axes[1].set_title('Train Loss')
    axes[1].set_xlabel('Evaluation Round')
    axes[1].legend(fontsize=9)

    plt.suptitle('Comparação entre Experimentos', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'plots', 'comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plot salvo: plots/comparison.png')
elif len(h5_files) == 1:
    print('Apenas 1 arquivo H5 disponível. Rode mais experimentos para comparar.')
else:
    print('Nenhum arquivo H5 encontrado.')
