import os, sys

# attacker = "selfish"
# gamma = 50
# adv_perc = 35

def gen(attacker, gamma, adv_perc):
    n = 100
    total_hp = 10000
    adv_hp = int(adv_perc / 100 * total_hp)
    honest_hps = [(total_hp - adv_hp) // (n-1) for i in range(n-2)] + [(total_hp-adv_hp) - (total_hp - adv_hp) // (n-1) * (n-2)]
    hfs = [i / total_hp for i in honest_hps] + [adv_hp / total_hp]

    print(gamma, adv_perc, sum(hfs), sum(honest_hps)+adv_hp)

    config = {
        "low_rho" : 10,
        "high_rho" : 500,
        "num_peers" : n,
        "slow_cij_val" : 5000,
        "high_cij_val" : 100000,
        "dij_cij_factor" : 96,
        "slow_fraction" : 0.5,
        "babasi_albert_m" : 3,
        "txn_inter_arrival_time" : 10000000,
        "net_mean_mining_time" : 600000,
        "hashing_fractions" : hfs,
        "mining_fee" : 20,
        "max_time" : 300000000,
        "text_files_folder" : '"Block_Arrivals"',
        "attacker" : f'"{attacker}"',
        "attacker_connection" : gamma / 100
    }
    path = os.path.join("configs", f"{attacker}", f"gamma_{gamma}", f"{attacker}_{adv_perc}_{gamma}.yaml")
    with open(path, 'w') as f:
        for k,v in config.items():
            if isinstance(v, list):
                print(f"{k} :", file=f)
                for e in v:
                    print(f"  - {e}", file=f)
            else:
                print(f"{k} : {v}", file=f)

if __name__ == "__main__":
    gammas = [25, 50, 75]
    hps = [10, 20, 35, 50]
    for gamma in gammas:
        for hp in hps:
            gen("stubborn", gamma, hp)