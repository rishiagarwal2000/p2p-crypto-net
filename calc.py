def chain_frac(alpha, gamma):
    frac = (alpha * (1-alpha)**2 * (4*alpha + gamma*(1-2*alpha)) - alpha**3) / (1 - alpha * (1+(2-alpha)*alpha))
    return frac

if __name__ == "__main__":
    for alpha in [0.1, 0.2, 0.35, 0.5]:
        print(f"{chain_frac(alpha, 0.25)}, {chain_frac(alpha, 0.5)}, {chain_frac(alpha, 0.75)}")