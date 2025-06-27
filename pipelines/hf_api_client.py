from huggingface_hub import InferenceClient

client = InferenceClient(
	provider="nebius",
	api_key="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"
)

doc = """
Name:    Hillary Gordon                   Unit No: 94
 
Admission Date: 2125-05-05 21:50:00              Discharge Date:    2125-05-06 13:03:00
 
Date of Birth:    2049-05-06             Sex:   M
 
Service: UROLOGY
 
Allergies: 
No Known Allergies / Adverse Drug Reactions
 
Attending:  Dr. Norma Fells.
 
Chief Complaint:
nephrolithiasis, acute kidney injury
 
Major Surgical or Invasive Procedure:
Cystoscopy, left ureteral stent placement.

 
History of Present Illness:
76  yo diabetic male, found to have at least 2 separate left 
ureteral stones, 4 mm at left UVJ and 6 mm at proximal ureter. 
His UA is unremarkable and he is without fevers. His creatinine 
is elevated to 1.4 on arrival and 1.5 on recheck after fluids. 
Discussed this with the patient, and ultimately recommended
cystoscopy and placement of left ureteral stent for 
decompression given his elevated creatinine.
 
Past Medical History:
Problems  (Last Verified - None on file):
DIABETES TYPE II                                                
NEPHROLITHIASIS                                                 

Surgical History  (Last Verified - None on file):
No Surgical History currently on file.   
 
Social History:
 
Family History:
No Family History currently on file.         

 
Physical Exam:
WdWn male, NAD, AVSS
Interactive, cooperative
Abdomen soft, Nt/Nd
Lower extremities w/out edema or pitting and no report of calf 
pain

 
Pertinent Results:
  10:36PM BLOOD WBC-10.1* RBC-5.54 Hgb-14.2 Hct-44.0 
MCV-79* MCH-25.6* MCHC-32.3 RDW-12.9 RDWSD-36.7 Plt  

  10:36PM BLOOD Neuts-64.3   Monos-6.9 Eos-2.9 
Baso-0.4 Im   AbsNeut-6.50* AbsLymp-2.49 AbsMono-0.70 
AbsEos-0.29 AbsBaso-0.04

  06:28AM BLOOD Glucose-193* UreaN-13 Creat-1.4* Na-143 
K-4.9 Cl-107 HCO3-24 AnGap-12
  05:39AM BLOOD Glucose-91 UreaN-15 Creat-1.5* Na-139 
K-4.8 Cl-102 HCO3-24 AnGap-13
  10:36PM BLOOD Glucose-260* UreaN-18 Creat-1.4* Na-135 
K-4.6 Cl-99 HCO3-18* AnGap-18

  10:36PM BLOOD ALT-23 AST-14 AlkPhos-93 TotBili-0.2

  06:28AM BLOOD Calcium-8.8 Mg-2.0

  10:36PM BLOOD Albumin-4.0

  03:16AM BLOOD Lactate-1.6

  12:35AM URINE Color-Yellow Appear-Clear Sp  
  12:35AM URINE Blood-SM* Nitrite-NEG Protein-TR* 
Glucose-1000* Ketone-NEG Bilirub-NEG Urobiln-NEG pH-6.0 
Leuks-NEG
  12:35AM URINE RBC-14* WBC-3 Bacteri-FEW* Yeast-NONE 
Epi-<1
  12:35AM URINE Mucous-RARE*

  01:05PM OTHER BODY FLUID STONE ANALYSIS-PND

  12:35 am URINE

                            **FINAL REPORT  

   URINE CULTURE (Final  : 
      ESCHERICHIA COLI.    10,000-100,000 CFU/mL. 
         PRESUMPTIVE IDENTIFICATION. 
         Cefazolin interpretative criteria are based on a dosage 
regimen of
         2g every 8h. 

                              SENSITIVITIES: MIC expressed in 
MCG/ML
                      
                   
                             ESCHERICHIA COLI
                             |   
AMPICILLIN------------     4 S
AMPICILLIN/SULBACTAM--   <=2 S
CEFAZOLIN-------------   <=4 S
CEFEPIME--------------   <=1 S
CEFTAZIDIME-----------   <=1 S
CEFTRIAXONE-----------   <=1 S
CIPROFLOXACIN---------<=0.25 S
GENTAMICIN------------   <=1 S
MEROPENEM-------------<=0.25 S
NITROFURANTOIN--------  <=16 S
PIPERACILLIN/TAZO-----   <=4 S
TOBRAMYCIN------------   <=1 S
TRIMETHOPRIM/SULFA----   <=1 S

 

 
Medications on Admission:
The Preadmission Medication list may be inaccurate and requires 
futher investigation.
1. MetFORMIN (Glucophage) 1000 mg PO BID 
2. GlipiZIDE 20 mg PO DAILY 
3. Januvia (SITagliptin) 100 mg oral DAILY 

 
Discharge Medications:
1.  Acetaminophen 650 mg PO Q6H:PRN Pain - Mild  
2.  Cephalexin 250 mg PO Q6H Duration: 7 Days 
RX *cephalexin 250 mg ONE tablet(s) by mouth Q6hrs Disp #*28 
Tablet Refills:*0 
3.  Docusate Sodium 100 mg PO BID 
RX *docusate sodium 100 mg ONE capsule(s) by mouth twice a day 
Disp #*60 Capsule Refills:*0 
4.  OxyCODONE (Immediate Release) 5 mg PO Q4H:PRN Pain - 
Moderate 
RX *oxycodone 5 mg ONE tablet(s) by mouth Q4hrs Disp #*10 Tablet 
Refills:*0 
5.  Pravastatin 80 mg PO DAILY  
6.  Sodium Bicarbonate 650 mg PO TID 
RX *sodium bicarbonate 650 mg ONE tablet(s) by mouth three times 
a day Disp #*28 Tablet Refills:*0 
7.  Tamsulosin 0.4 mg PO QHS 
RX *tamsulosin 0.4 mg ONE capsule(s) by mouth DAILY Disp #*30 
Capsule Refills:*0 
8.  amLODIPine 10 mg PO DAILY  
9.  Aspirin 81 mg PO DAILY  
10.  GlipiZIDE 20 mg PO DAILY  
11.  Januvia (SITagliptin) 100 mg oral DAILY  
12.  MetFORMIN (Glucophage) 1000 mg PO BID  
13.Outpatient Lab Work
Please have repeat lab work (Chem 7) through your PCP     
days after discharge (to check your kidney function). Call to 
arrange when you get home today. 

 
Discharge Disposition:
Home
 
Discharge Diagnosis:
nephrolithiasis; Obstructing left ureteral stones
acute kidney injury
urinary tract infection (E.Coli)

 
Discharge Condition:
Mental Status: Clear and coherent.
Level of Consciousness: Alert and interactive.
Activity Status: Ambulatory - Independent.

 
Discharge Instructions:
-You can expect to see occasional blood in your urine and to 
possibly experience some urgency and frequency over the next 
month;  this may be related to the passage of stone fragments or 
the indwelling ureteral stent.

-The kidney stone may or may not have been removed AND/or there 
may fragments/others still in the process of passing.

-You may experience some pain associated with spasm of your 
ureter.; This is normal. Take the narcotic pain medication as 
prescribed if additional pain relief is needed.

-Ureteral stents MUST be removed or exchanged and therefore it 
is IMPERATIVE that you follow-up as directed. 

-Do not lift anything heavier than a phone book (10 pounds) 

-You may continue to periodically see small amounts of blood in 
your urine--this is normal and will gradually improve

-Resume your pre-admission/home medications EXCEPT as noted. You 
should ALWAYS call to inform, review and discuss any medication 
changes and your post-operative course with your primary care 
doctor. 

-For pain control, try TYLENOL FIRST, then ibuprofen, and then 
take the narcotic pain medication as prescribed if additional 
pain relief is needed.

-You may be given prescriptions for a stool softener and/or a 
gentle laxative. These are  over-the-counter medications that 
may be health care spending account reimbursable. 

-Colace (docusate sodium) may have been prescribed to avoid 
post-surgical constipation or constipation related to use of 
narcotic pain medications. Discontinue if loose stool or 
diarrhea develops. Colace is a stool-softener, NOT a laxative.

-Senokot (or any gentle laxative) may have been prescribed to 
further minimize your risk of constipation. 

-Do not eat constipating foods for 58  weeks, drink plenty of 
fluids to keep hydrated
 
Followup Instructions:
"""

messages = [
	{
		"role": "user",
		"content": f"""
You are a medical expert. Convert the following medical record into a summary. You must not reveal any private information, such as names, dates and location. 
{doc}
"""
	}
]

completion = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct", 
	messages=messages, 
	max_tokens=500,
)

print(completion.choices[0].message)


