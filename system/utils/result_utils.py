import h5py
import numpy as np
import os


def average_data(args, times=10):
    """Computes and prints statistics across multiple experiment runs.
    
    Args:
        args: The argparse namespace with all experiment configuration.
        times: Number of experiment repetitions to aggregate.
    """
    test_acc_all = []
    test_auc_all = []
    train_loss_all = []
    round_time_all = []

    for i in range(times):
        file_name = _build_experiment_name(args, i)
        data = read_h5_results(file_name)
        test_acc_all.append(np.array(data.get('rs_test_acc', [])))
        test_auc_all.append(np.array(data.get('rs_test_auc', [])))
        train_loss_all.append(np.array(data.get('rs_train_loss', [])))
        # Support both new key (rs_round_time) and old key (rs_train_time)
        round_time_all.append(np.array(data.get('rs_round_time', data.get('rs_train_time', []))))

    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)

    max_accuracy = [acc.max() for acc in test_acc_all if len(acc) > 0]
    if max_accuracy:
        print("std for best accuracy:", np.std(max_accuracy))
        print("mean for best accuracy:", np.mean(max_accuracy))

    max_auc = [auc.max() for auc in test_auc_all if len(auc) > 0]
    if max_auc:
        print("std for best AUC:", np.std(max_auc))
        print("mean for best AUC:", np.mean(max_auc))

    min_loss = [loss.min() for loss in train_loss_all if len(loss) > 0]
    if min_loss:
        print("std for min loss:", np.std(min_loss))
        print("mean for min loss:", np.mean(min_loss))

    avg_time = [t.mean() for t in round_time_all if len(t) > 0]
    if avg_time:
        print("mean round time (s):", np.mean(avg_time))

    print("=" * 50)


def _build_experiment_name(args, times):
    """Builds the experiment name matching the server's naming convention.
    
    This MUST produce the exact same string as Server._build_experiment_name()
    in serverbase.py for file lookup to work correctly.
    """
    model_str = getattr(args, 'model_str', 'unknown')
    atk = getattr(args, 'atack', 'none')
    run_id = getattr(args, 'run_id', '')

    name = (
        f"{args.dataset}_{args.algorithm}"
        f"_cc{args.cluster_comparation}"
        f"_nmal{args.n_client_malicious}"
        f"_rfake{args.rate_client_fake}"
        f"_atk-{atk}"
        f"_m-{model_str}"
        f"_nc{args.num_clients}"
        f"_gr{args.global_rounds}"
        f"_le{args.local_epochs}"
        f"_lr{args.local_learning_rate}"
        f"_{args.goal}"
        f"_t{times}"
        f"_{run_id}"
    )
    return name


def read_h5_results(file_name):
    """Reads all datasets from an HDF5 results file.

    Returns a dictionary with dataset names as keys and numpy arrays as values.
    """
    file_path = "../results/" + file_name + ".h5"

    data = {}
    with h5py.File(file_path, 'r') as hf:
        for key in hf.keys():
            data[key] = np.array(hf.get(key))

    print(f"Read {file_path}: {list(data.keys())}, lengths: {[len(v) for v in data.values()]}")
    return data