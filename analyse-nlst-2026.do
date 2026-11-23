version 16.1

capture log close
clear frames

log using "analyse-nslt-2026", replace

set seed 2187

import sas using "C:\Users\tms211\Downloads\package-nlst-519.2026-09-24\participant_d021326.sas7bdat", clear

frame create lungcancer
frame lungcancer: import sas using "C:\Users\tms211\Downloads\package-nlst-519.2026-09-24\Lung Cancer\lung_cancer_d021326.sas7bdat"
frame lungcancer: keep if lc_order == 1 // We are only interested in their first lung cancer
frlink 1:1 pid, frame(lungcancer)

// Calculate PLCO linear predictor
generate bmi = (weight / (height ^ 2)) * 703
generate plco = -4.532506
generate intensity_trans = (smokeday / 10) ^ (-1) - 0.4021541613
generate quit_duration = age - age_quit if cigsmok == 0
replace  quit_duration = 0 if missing(quit_duration)

replace plco = plco + 0.0778868 * (age - 62)
replace plco = plco - 0.7434744 if ethnic == 1
replace plco = plco - 0.0812744 * (max(2, educat) - 5) if educat <= 7
replace plco = plco - 0.0274194 * (bmi - 27) if !missing(bmi)
replace plco = plco + 0.3553063 if diagcopd == 1
replace plco = plco + 0.4589971 if cancblad | cancbrea | canccerv | canccolo | cancesop | canckidn | canclary | canclung | cancnasa | cancoral | cancpanc | cancphar | cancstom | cancthyr | canctran
replace plco = plco + 0.5871850 if fambrother | famchild | famfather | fammother | famsister
replace plco = plco + 0.2597431 if cigsmok
replace plco = plco - 1.8226060 * intensity_trans
replace plco = plco + 0.0317321 * (smokeyr - 27)
replace plco = plco - 0.0308572 * (quit_duration - 10)

generate plco_predrisk = 1/(1+exp(-plco))

// Calculate comprehensive smoking index (without lag)
generate csindex = ln(smokeday + 1) * (1 - 0.5 ^ (smokeyr / 26))
replace  csindex = csindex * 0.5 ^ (quit_duration / 26)

// Calculate LLP linear predictor
tempvar x xp5 y llp_int1 llp_int2
generate `x' = 5 * floor(age / 5)
generate `xp5' = `x' + 5
generate `y' = age - `x'

capture frame drop intercepts
frame create intercepts
frame change intercepts

input gender age intercept
1 40 -9.84
1 45 -8.94
1 50 -8.09
1 55 -7.41
1 60 -6.75
1 65 -6.34
1 70 -6.09
1 75 -5.61
1 80 -5.46
2 40 -10.37
2 45 -8.53
2 50 -7.93
2 55 -6.97
2 60 -6.69
2 65 -6.46
2 70 -5.96
2 75 -5.70
2 80 -5.89
end
/*
 * - LLPv2 -
input gender age intercept
1 40 -9.06
1 45 -8.16
1 50 -7.31
1 55 -6.63
1 60 -5.97
1 65 -5.56
1 70 -5.31
1 75 -4.83
1 80 -4.68
2 40 -9.90
2 45 -8.06
2 50 -7.46
2 55 -6.50
2 60 -6.22
2 65 -5.99
2 70 -5.49
2 75 -5.23
2 80 -5.42
end
*/

frame change default

frlink m:1 gender `x', frame(intercepts gender age) generate(intercept1)
frlink m:1 gender `xp5', frame(intercepts gender age) generate(intercept2)
frget `llp_int1' = intercept, from(intercept1)
frget `llp_int2' = intercept, from(intercept2)

generate llp = ((5 - `y' - 0.5) * `llp_int1' + (`y' + 0.5) * `llp_int2') / 5

frame drop intercepts
drop intercept1
drop intercept2

replace llp = llp + 0.7692 if smokeyr <= 20
replace llp = llp + 1.4516 if smokeyr > 20 & smokeyr <= 40
replace llp = llp + 2.5072 if smokeyr > 40 & smokeyr <= 60
replace llp = llp + 2.7243 if smokeyr > 60
replace llp = llp + 0.6025 if diagpneu
replace llp = llp + 0.6343 if wrkasbe
replace llp = llp + 0.6754 if cancblad | cancbrea | canccerv | canccolo | cancesop | canckidn | canclary | canclung | cancnasa | cancoral | cancpanc | cancphar | cancstom | cancthyr | canctran
replace llp = llp + 0.7034 if fambrother | famchild | famfather | fammother | famsister

/**
 * CALCULATE 8th EDITION STAGE
 */
frame change lungcancer

// Link m:1 even though we have removed all but first lung cancer for each
// participant in case we decide not to remove subsequent lung cancers
frlink m:1 pid, frame(default)

label define stag_6thed 110 "IA" 120 "IB" 210 "IIA" 220 "IIB" 310 "IIIA" 320 "IIIB" 400 "IV" 900 "Occult carcinoma" 994 "Carcinoid, cannot be assessed" 999 "Unknown, cannot be assessed" 888 "TNM not available"
label define stag_7thed 110 "IA" 120 "IB" 210 "IIA" 220 "IIB" 310 "IIIA" 320 "IIIB" 400 "IV" 900 "Occult carcinoma" 999 "Unknown, cannot be assessed" 888 "TNM not available"
label define t_7thed 999 "Tx" 110 "T1a" 120 "T1b" 200 "T2" 210 "T2a" 220 "T2b" 300 "T3" 400 "T4"
label define n_7thed 999 "Nx" 0 "N0" 100 "N1" 200 "N2" 300 "N3"
label define m_7thed 999 "Mx" 0 "M0" 100 "M1" 110 "M1a" 120 "M1b"
label define source_best_stage 1 "Pathological" 2 "Clinical" 3 "Mixture" 5 "Reporting stage only" 6 "Stage cannot be assessed" 94 "Carcinoid, stage cannot be assessed" 98 "TNM not avilable" 99 "Missing TNM"

label values de_stag stag_6thed
label values de_stag_7thed stag_7thed
label values *_t_7thed t_7thed
label values *_n_7thed n_7thed
label values *_m_7thed m_7thed
label values source_best_stage source_best_stage

generate t_7thed = path_t_7thed
replace  t_7thed = clinical_t_7thed if missing(path_t_7thed) | path_t_7thed == "Tx":t_7thed
generate n_7thed = path_n_7thed
replace  n_7thed = clinical_n_7thed if missing(path_n_7thed) | path_n_7thed == "Nx":n_7thed
replace  n_7thed = "N0":n_7thed if n_7thed == "Nx":n_7thed & (de_stag_7thed <= 4 | (de_stag_7thed == "IIA":stag_7thed & t_7thed == "T2b":t_7thed) | (de_stag_7thed == "IIB":stag_7thed & t_7thed == "T3":t_7thed) | (de_stag_7thed == "IIIA":stag_7thed & t_7thed == "T4":t_7thed))
generate m_7thed = path_m_7thed
replace  m_7thed = clinical_m_7thed if missing(path_m_7thed) | path_m_7thed == "Mx":m_7thed
replace  m_7thed = "M0":m_7thed if m_7thed == "Mx":m_7thed & de_stag_7thed < 10

label values t_7thed t_7thed
label values n_7thed n_7thed
label values m_7thed m_7thed

egen tnm_7thed = concat(t_7thed n_7thed m_7thed), decode p(" ")

label define t_8thed 1 "Tx" 2 "T1a" 3 "T1b" 4 "T1c" 5 "T2a" 6 "T2b" 7 "T3" 8 "T4"

generate t_8thed = 1
label values t_8thed t_8thed
replace t_8thed = "T1b":t_8thed if t_7thed == "T1a":t_7thed
replace t_8thed = "T1a":t_8thed if t_7thed == "T1a":t_7thed & lesionsize <= 10
replace t_8thed = "T1c":t_8thed if t_7thed == "T1b":t_7thed
replace t_8thed = "T3":t_8thed if t_7thed == "T2a":t_7thed | t_7thed == "T2b":t_7thed | t_7thed == "T2":t_7thed
replace t_8thed = "T2a":t_8thed if t_7thed == "T2a":t_7thed & lesionsize <= 40
replace t_8thed = "T2b":t_8thed if t_7thed == "T2a":t_7thed & lesionsize > 40 & lesionsize <= 50
replace t_8thed = "T4":t_8thed if t_7thed == "T3":t_7thed | t_7thed == "T4":t_7thed
replace t_8thed = "T3":t_8thed if t_7thed == "T3":t_7thed & lesionsize <= 70

label define stag_custom8 1 "IA1" 2 "IA2" 3 "IA3" 4 "IB" 5 "II" 6 "III" 7 "IV"

generate stag_custom8 = "IV":stag_custom8 if de_stag_7thed <= "IV":stag_7thed
label values stag_custom8 stag_custom8
replace stag_custom8 = "IA1":stag_custom8 if de_stag_7thed ==  "IA":stag_7thed & t_8thed == "T1a":t_8thed
replace stag_custom8 = "IA2":stag_custom8 if de_stag_7thed ==  "IA":stag_7thed & t_8thed == "T1b":t_8thed
replace stag_custom8 = "IA3":stag_custom8 if de_stag_7thed ==  "IA":stag_7thed & t_8thed == "T1c":t_8thed
replace stag_custom8 =  "IB":stag_custom8 if de_stag_7thed ==  "IB":stag_7thed & t_8thed == "T2a":t_8thed
replace stag_custom8 =  "II":stag_custom8 if de_stag_7thed ==  "IB":stag_7thed & t_8thed == "T2b":t_8thed
replace stag_custom8 =  "II":stag_custom8 if de_stag_7thed ==  "IB":stag_7thed & t_8thed == "T3":t_8thed
replace stag_custom8 =  "II":stag_custom8 if de_stag_7thed == "IIA":stag_7thed | de_stag_7thed == "IIB":stag_7thed
replace stag_custom8 = "III":stag_custom8 if de_stag_7thed == "IIB":stag_7thed & ((t_8thed == "T3":t_8thed & n_7thed == "N1":n_7thed) | (t_8thed == "T4":t_8thed & n_7thed == "N0":n_7thed))
replace stag_custom8 = "III":stag_custom8 if de_stag_7thed == "IIIA":stag_7thed | de_stag_7thed == "IIIB":stag_7thed
replace stag_custom8 = "III":stag_custom8 if de_stag_7thed > "IV":stag_7thed & de_stag <= "IV":stag_6thed
replace stag_custom8 = "IA3":stag_custom8 if de_stag_7thed > "IV":stag_7thed & de_stag == "IA":stag_6thed
replace stag_custom8 =  "IV":stag_custom8 if de_stag_7thed > "IV":stag_7thed & de_stag == "IV":stag_6thed

frget contactstatus fup_days, from(default)

// NOTE this will be survival from diagnosis of FIRST lung cancer
generate surv_days = fup_days - candx_days

stset surv_days, failure(contactstatus == 2) scale(365)
stpmean, at(7)
mlogit stag_custom8 pseudo
predict p_stag_custom8_*
replace stag_custom8 = 1 + irecode(uniform(), ///
	p_stag_custom8_1, ///
	p_stag_custom8_1 + p_stag_custom8_2, ///
	p_stag_custom8_1 + p_stag_custom8_2 + p_stag_custom8_3, ///
	p_stag_custom8_1 + p_stag_custom8_2 + p_stag_custom8_3 + p_stag_custom8_4, ///
	p_stag_custom8_1 + p_stag_custom8_2 + p_stag_custom8_3 + p_stag_custom8_4 + p_stag_custom8_5, ///
	p_stag_custom8_1 + p_stag_custom8_2 + p_stag_custom8_3 + p_stag_custom8_4 + p_stag_custom8_5 + p_stag_custom8_6) ///
	if missing(stag_custom8)

// Handle SCLC

gen sclc = inlist(lc_morph, 8002, 8041, 8042, 8043, 8044, 8045)

label define valcsg 1 "Limited" 2 "Extensive" 3 "Not available"
label values valcsg valcsg

gen stag_sclc = valcsg if sclc
replace stag_sclc = "Limited":valcsg if sclc & stag_custom8 < "IIIB":stag_custom8
replace stag_sclc = "Extensive":valcsg if sclc & (stag_custom8 == "IIIB":stag_custom8 | stag_custom8 == "IV":stag_custom8)
label values stag_sclc valcsg

frame change default

// Pull data needed
frget stag_custom8 sclc stag_sclc, from(lungcancer)

// Save
save trial.dta, replace

log close

