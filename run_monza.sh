#!/bin/bash
# rush.sh - Executa os 5 experimentos em sequência
# Uso: bash rush.sh

BASE_ARGS="-nc 100 -jr 1.0 -gr 150 -data Cifar10 -t 2 -ls 1 -lbs 10 -lr 0.005 -m CNN -dev cpu"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="../results/logs"
mkdir -p "$LOG_DIR"

echo "============================================"
echo " Iniciando bateria de experimentos"
echo " Data: $(date)"
echo "============================================"

# -------------------------------------------------
# 1. MONZA Original (cc=3 - padrão do repositório)
# -------------------------------------------------
echo ""
echo "[1/5] MONZA Original (cc=3)"
echo "----------------------------------------"
python main.py -nmc 30 $BASE_ARGS -atk all -cc 3 -rfake 1 \
    2>&1 | tee "$LOG_DIR/1_monza_cc3_${TIMESTAMP}.log"
echo "MONZA Original concluído. Exit code: $?"

# -------------------------------------------------
# 2. MONZA Multicritério (cc=6 - nova proposta)
# -------------------------------------------------
echo ""
echo "[2/5] MONZA - Multicritério (cc=6)"
echo "----------------------------------------"
python main.py -nmc 30 $BASE_ARGS -atk all -cc 6 -rfake 1 \
    2>&1 | tee "$LOG_DIR/2_monza_cc6_${TIMESTAMP}.log"
echo "MONZA Multicritério concluído. Exit code: $?"

# -------------------------------------------------
# 3. zPROBE (Baseline - cc=2 clusterização)
# -------------------------------------------------
echo ""
echo "[3/5] zPROBE - Clusterização (cc=2)"
echo "----------------------------------------"
python main.py -nmc 30 $BASE_ARGS -atk all -cc 2 -rfake 1 \
    2>&1 | tee "$LOG_DIR/3_zprobe_cc2_${TIMESTAMP}.log"
echo "zPROBE concluído. Exit code: $?"

# -------------------------------------------------
# 4. Default FL Sem Defesa (cc=5)
# -------------------------------------------------
echo ""
echo "[4/5] Default FL - Sem defesa (cc=5)"
echo "----------------------------------------"
python main.py -nmc 30 $BASE_ARGS -atk all -cc 5 -rfake 1 \
    2>&1 | tee "$LOG_DIR/4_nodefense_cc5_${TIMESTAMP}.log"
echo "Sem defesa concluído. Exit code: $?"

# -------------------------------------------------
# 5. Default FL (Benchmark Ideal - Sem Ataques)
# -------------------------------------------------
echo ""
echo "[5/5] Benchmark Ideal - Sem ataques (nmc=0)"
echo "----------------------------------------"
python main.py -nmc 0 $BASE_ARGS -cc 5 \
    2>&1 | tee "$LOG_DIR/5_benchmark_nmc0_${TIMESTAMP}.log"
echo "Benchmark concluído. Exit code: $?"

echo ""
echo "============================================"
echo " Todos os experimentos concluídos!"
echo " Logs salvos em: $LOG_DIR"
echo "============================================"
