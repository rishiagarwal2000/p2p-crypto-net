import os, sys

if __name__ == "__main__":
    gammas = [25, 50, 75]
    hps = [10, 20, 35, 50]
    attacker = "stubborn"
    for gamma in gammas:
        for hp in hps:
            print("#"*20 + f"  {gamma} {hp} " + "#"*20)
            path = os.path.join("configs", f"{attacker}", f"gamma_{gamma}", f"{attacker}_{hp}_{gamma}.yaml")
            cmd = f"python p2p_code.py --config {path} --graph_seed 0"
            os.system(cmd)
