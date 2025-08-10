# Compute EM for 10 iterations and print a combined markdown table for direct paste.
import math, pandas as pd
trials = [(3,7),(4,6),(7,3),(9,1)]
thetaA = 0.7
thetaB = 0.2
iterations = 10

rows = []
thetahistory = []

def likelihood(theta, H, T):
    return (theta**H) * ((1-theta)**T)

for it in range(1, iterations+1):
    totals_A_heads = totals_A_tails = totals_B_heads = totals_B_tails = 0.0
    for t_index, (H, T) in enumerate(trials, start=1):
        likA = likelihood(thetaA, H, T)
        likB = likelihood(thetaB, H, T)
        denom = likA + likB
        gamma = likA/denom if denom>0 else 0.5
        gammaB = 1 - gamma
        exp_H_A = H * gamma
        exp_T_A = T * gamma
        exp_H_B = H * gammaB
        exp_T_B = T * gammaB
        totals_A_heads += exp_H_A
        totals_A_tails += exp_T_A
        totals_B_heads += exp_H_B
        totals_B_tails += exp_T_B
        rows.append({
            "Iteration": it,
            "Trial": t_index,
            "H": H,
            "T": T,
            "likelihood_A": likA,
            "likelihood_B": likB,
            "P(A|data)": gamma,
            "P(B|data)": gammaB,
            "exp_H_A": exp_H_A,
            "exp_T_A": exp_T_A,
            "exp_H_B": exp_H_B,
            "exp_T_B": exp_T_B,
            "thetaA_before": thetaA,
            "thetaB_before": thetaB
        })
    total_flips_A = totals_A_heads + totals_A_tails
    total_flips_B = totals_B_heads + totals_B_tails
    new_thetaA = totals_A_heads / total_flips_A if total_flips_A>0 else thetaA
    new_thetaB = totals_B_heads / total_flips_B if total_flips_B>0 else thetaB
    thetahistory.append({
        "Iteration": it,
        "thetaA_before": thetaA,
        "thetaB_before": thetaB,
        "exp_heads_A_total": totals_A_heads,
        "exp_tails_A_total": totals_A_tails,
        "exp_heads_B_total": totals_B_heads,
        "exp_tails_B_total": totals_B_tails,
        "thetaA_after": new_thetaA,
        "thetaB_after": new_thetaB
    })
    thetaA, thetaB = new_thetaA, new_thetaB

df = pd.DataFrame(rows)
df_round = df.round(6)
# Create markdown table
md = df_round.to_markdown(index=False)
summary = pd.DataFrame(thetahistory).round(6).to_markdown(index=False)

print("## EM detailed table (Iterations 1-10, 4 trials each)\n")
print(md)
print("\n## Summary (theta before/after each iteration)\n")
print(summary)
