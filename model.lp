% --- 1. SELECTION LOGIC ---
{ select(S, W) } :- primary(S), scenario(W).
{ select_backup(B, W) } :- backup(B), scenario(W).

% --- 2. CAPACITY & DISRUPTION ---
eff_cap(S, W, 0) :- disrupted(S, W), primary(S).
eff_cap(S, W, C) :- cap_primary(S, C), not disrupted(S, W), scenario(W).

% --- 3. RESILIENCE: DEMAND SATISFACTION ---
:- scenario(W), demand(D),
   #sum{ C,S : select(S, W), eff_cap(S, W, C); 
         CB,B : select_backup(B, W), cap_backup(B, CB) } < D.

% --- 4. NATURAL BACKUP NEED TRIGGER ---
total_avail_primary_cap(W, Total) :- scenario(W),
    Total = #sum{ C,S : primary(S), eff_cap(S, W, C) }.

total_avail_primary_soc(W, Total) :- scenario(W),
    Total = #sum{ SS,S : primary(S), soc_primary(S, SS) }.

backup_needed(W) :- total_avail_primary_cap(W, Total), demand(D), Total < D.
backup_needed(W) :- total_avail_primary_soc(W, Total), soc_threshold(T), Total < T.

:- backup_needed(W), not 1 { select_backup(B, W) : backup(B) }.
:- not backup_needed(W), select_backup(B, W).

% --- 5. SUSTAINABILITY THRESHOLDS ---
:- scenario(W), soc_threshold(T),
    #sum{ SS,S : select(S, W), soc_primary(S, SS); 
          SB,B : select_backup(B, W), soc_backup(B, SB) } < T.

:- scenario(W), env_threshold(T),
    #sum{ ES,S : select(S, W), env_primary(S, ES); 
          EB,B : select_backup(B, W), env_backup(B, EB) } < T.

% --- 6. OPTIMIZATION (CORRECTED SYNTAX) ---
% Priority 2: Minimize Backup Costs (Financial Selectivity)
#minimize { CB@2,B,W : select_backup(B, W), cost_backup(B, CB) }.

% Priority 1: Maximize Sustainability (ESG Quality)
#minimize { -SS@1,S,W : select(S, W), soc_primary(S, SS) }.
#minimize { -SB@1,B,W : select_backup(B, W), soc_backup(B, SB) }.

% --- 7. PERFORMANCE BOOSTER (Symmetry Breaking) ---
% This prevents the "Stuck" issue by forcing the solver to pick b01 before b02, etc.
:- select_backup(B2, W), B1 < B2, backup(B1), backup(B2), not select_backup(B1, W).

#show select/2.
#show select_backup/2.