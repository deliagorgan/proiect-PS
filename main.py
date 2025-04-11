import pandas as pd
import seaborn as sns
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from scipy import stats as scipy_stats
from scipy.stats import chi2_contingency
from cluster import  page_analiza_clustering_diabet
from regresie import page_model_regresie_diabet
st.set_page_config(page_title="Analiza datelor despre diabet", page_icon="📊", layout="wide")


@st.cache_data
def load_data():
    df_diabet = pd.read_csv("./data/diabetes_data.csv")
    return df_diabet


df_diabet = load_data()

st.markdown('<h1 style="color: inherit; font-size: 40px; text-align: center;">Analiza datelor despre diabet</h1>',
            unsafe_allow_html=True)

section = st.sidebar.radio("Navigați la:",
                           ["Detalii despre setul de date", "Analiza Exploratorie", "Analiza Clustering K-means", "Model de Regresie Liniară"])

# ---------------------------
# Secțiunea: Detalii despre setul de date
# ---------------------------
if section == "Detalii despre setul de date":
    st.write(df_diabet)

    st.subheader("Variabilele setului de date:\n")
    try:
        with open("explicare_variabile.txt", "r") as file:
            continut = file.read()
        st.markdown(continut)
    except FileNotFoundError:
        st.markdown("""
        ## Descrierea variabilelor din setul de date:

        - **PatientID**: ID-ul unic al pacientului
        - **Gender**: Genul pacientului
        - **Age**: Vârsta pacientului
        - **Ethnicity**: Originea etnică a pacientului
        - **Occupation**: Ocupația pacientului
        - **BMI**: Indicele de masă corporală
        - **AlcoholConsumption**: Nivelul de consum de alcool
        - **PhysicalActivity**: Nivelul de activitate fizică (în minute per săptămână)
        - **DietQuality**: Calitatea dietei pe o scală de la 1 la 10
        - **SleepQuality**: Calitatea somnului pe o scală de la 1 la 10
        - **SystolicBP**: Tensiunea arterială sistolică
        - **DiastolicBP**: Tensiunea arterială diastolică
        - **FastingBloodSugar**: Glicemia à jeun (mg/dL)
        - **HbA1c**: Hemoglobina glicată (%)
        - **SerumCreatinine**: Creatinină serică (mg/dL)
        - **BUNLevels**: Nivelul ureei sanguine (mg/dL)
        - **CholesterolTotal**: Colesterol total (mg/dL)
        - **CholesterolLDL**: Colesterol LDL (mg/dL)
        - **CholesterolHDL**: Colesterol HDL (mg/dL)
        - **CholesterolTriglycerides**: Trigliceride (mg/dL)
        - **FatigueLevels**: Niveluri de oboseală pe o scală de la 1 la 10
        - **MedicalCheckupsFrequency**: Frecvența controalelor medicale pe an
        - **MedicationAdherence**: Aderența la medicație pe o scală de la 1 la 10
        - **HealthLiteracy**: Nivelul de cunoștințe medicale pe o scală de la 1 la 10
        - **FamilyHistory**: Istorie familială de diabet (0 = Nu, 1 = Da)
        - **AntihypertensiveMedications**: Utilizare de medicamente antihipertensive (0 = Nu, 1 = Da)
        - **Statins**: Utilizare de statine (0 = Nu, 1 = Da)
        - **AntidiabeticMedications**: Utilizare de medicamente antidiabetice (0 = Nu, 1 = Da)
        - **DoctorInCharge**: ID-ul medicului responsabil
        - **Diagnosis**: Diagnosticul de diabet (0 = Nu, 1 = Da)
        """)

    numar_linii_initial = df_diabet.shape[0]
    df_diabet = df_diabet.drop_duplicates()
    numar_linii_dupa = df_diabet.shape[0]

    if numar_linii_initial == numar_linii_dupa:
        st.success("Setul de date nu conține înregistrări duplicate!")
    else:
        st.warning(
            f"Au fost eliminate {numar_linii_initial - numar_linii_dupa} înregistrări duplicate. Acum setul conține {numar_linii_dupa} înregistrări.")

    nan_summary = df_diabet.isnull().sum()
    if nan_summary.sum() > 0:
        st.warning("Setul de date conține valori lipsă:")
        st.write(nan_summary[nan_summary > 0])
    else:
        st.success("Setul de date nu conține valori lipsă!")

elif section == "Analiza Clustering K-means":
    page_analiza_clustering_diabet(df_diabet)
# ---------------------------
# Secțiunea: Analiza Exploratorie
# ---------------------------
# ---------------------------
# Secțiunea: Model de Regresie Liniară
# ---------------------------
elif section == "Model de Regresie Liniară":
    page_model_regresie_diabet(df_diabet)
elif section == "Analiza Exploratorie":
    st.markdown("### Curățarea datelor inițiale")
    st.markdown(
        "Am eliminat coloanele care nu influențează analiza, adică PatientID și DoctorInCharge, deoarece acestea sunt identificatori și nu conțin informații relevante pentru analiza noastră.")
    cols_to_drop = ['PatientID', 'DoctorInCharge']
    df_diabet = df_diabet.drop(cols_to_drop, axis=1)
    st.write(df_diabet.head())

    coloane_numerice = ["Age", "BMI", "AlcoholConsumption", "PhysicalActivity", "DietQuality", "SleepQuality",
                        "SystolicBP", "DiastolicBP", "FastingBloodSugar", "HbA1c", "SerumCreatinine",
                        "BUNLevels", "CholesterolTotal", "CholesterolLDL", "CholesterolHDL", "CholesterolTriglycerides",
                        "FatigueLevels", "MedicalCheckupsFrequency", "MedicationAdherence", "HealthLiteracy"]

    coloane_categorice = df_diabet.columns.difference(coloane_numerice).tolist()

    df_poz_diabet = df_diabet[df_diabet['Diagnosis'] == 1]
    df_neg_diabet = df_diabet[df_diabet['Diagnosis'] == 0]

    tab_analiza_generala, tab_outlieri, tab_analiza_distributie, tab_grafic_pairplot, tab_analiza_corelatii, tab_encodare, tab_standardizare, tab_normalizare = st.tabs(
        ["Analiza generală", "Analiza valori outlier", "Analiza distribuției datelor", "Grafic pairplot",
         "Analiza corelațiilor dintre variabile", "Encodarea variablelor discrete",
         "Standardizarea setului de date", "Normalizarea setului de date"])

    # ---------------------------
    # Tab: Analiza generală
    # ---------------------------
    with tab_analiza_generala:
        st.markdown(
            "<h2 style='text-align: center; color: blue;'>Analiza generală a datelor numerice pentru pacienți</h2>",
            unsafe_allow_html=True
        )
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                "<h3 style='text-align: center; color: red;'>Diagnosticați cu diabet</h3>",
                unsafe_allow_html=True
            )
            st.write(df_poz_diabet[coloane_numerice].describe())

        with col2:
            st.markdown(
                "<h3 style='text-align: center; color: green;'>Nediagnosticați cu diabet</h3>",
                unsafe_allow_html=True
            )
            st.write(df_neg_diabet[coloane_numerice].describe())

        st.markdown(
            "<h2 style='text-align: center; color: red;'>Interpretarea setului de date</h2>",
            unsafe_allow_html=True
        )
        st.markdown(
            f""" 
            Acest set de date conține **{len(df_diabet)} de observații** și măsoară diverse variabile legate de sănătatea pacienților care fie suferă, fie nu de diabet.

            **Distribuția diagnosticului:** {df_diabet['Diagnosis'].value_counts()[1]} pacienți cu diabet și {df_diabet['Diagnosis'].value_counts()[0]} pacienți fără diabet.

            ---
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            ### Interpretarea statisticilor descriptive:

            Comparând statisticile descriptive între pacienții diagnosticați cu diabet și cei fără diabet, putem observa diferențe importante:

            #### Factori cu diferențe semnificative:

            1. **Vârsta (Age)**: Pacienții cu diabet au o vârstă medie mai mare, ceea ce confirmă că riscul de diabet crește cu vârsta.

            2. **BMI**: Valoarea medie a indicelui de masă corporală este mai mare la pacienții cu diabet, sugerând o legătură între obezitate și diabet.

            3. **FastingBloodSugar și HbA1c**: După cum era de așteptat, pacienții cu diabet au valori semnificativ mai mari ale glicemiei à jeun și hemoglobinei glicate, acestea fiind criterii diagnostice pentru diabet.

            4. **Tensiunea arterială (SystolicBP și DiastolicBP)**: Pacienții cu diabet tind să aibă valori mai mari ale tensiunii arteriale, ceea ce confirmă asocierea frecventă între diabet și hipertensiune.

            5. **Colesterol (CholesterolTotal, CholesterolLDL, CholesterolTriglycerides)**: Valorile medii sunt mai ridicate la pacienții cu diabet, confirmând asocierea dintre diabet și dislipidemie.

            #### Factori cu diferențe moderate:

            1. **PhysicalActivity**: Pacienții fără diabet tind să aibă niveluri mai ridicate de activitate fizică, sugerând rolul protector al exercițiului fizic.

            2. **DietQuality**: Calitatea dietei este, în medie, mai bună la pacienții fără diabet, subliniind importanța alimentației în prevenirea diabetului.

            3. **SleepQuality**: Pacienții fără diabet raportează o calitate mai bună a somnului, ceea ce poate fi atât o cauză, cât și o consecință a stării de sănătate.

            #### Interpretare generală:

            Aceste statistici descriptive confirmă factorii de risc cunoscuți pentru diabetul zaharat: vârsta înaintată, excesul ponderal, sedentarismul, alimentația neadecvată și prezența altor afecțiuni metabolice (hipertensiune, dislipidemie). 

            Diferențele observate susțin abordarea multifactorială în prevenția și managementul diabetului, incluzând intervenții legate de stil de viață (activitate fizică, alimentație, somn) și monitorizarea factorilor de risc cardiovascular asociați.
            """
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x='Diagnosis', data=df_diabet, palette=['green', 'red'])
        ax.set_title('Distribuția diagnosticului de diabet')
        ax.set_xlabel('Diagnostic (0 = Nu, 1 = Da)')
        ax.set_ylabel('Număr de pacienți')
        for p in ax.patches:
            ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom')
        st.pyplot(fig)

    # ---------------------------
    # Tab: Analiza Outlieri
    # ---------------------------
    with tab_outlieri:
        st.markdown(
            "<h2 style='text-align: center; color: red;'>Analiza valorilor extreme pentru toate valorile numerice</h2>",
            unsafe_allow_html=True
        )

        st.markdown(
            """
            ### Interpretarea valorilor extreme (outlier):

            Valorile extreme (outlieri) pot avea un impact semnificativ asupra analizei datelor și asupra modelelor de machine learning. Identificarea și tratarea acestora este un pas important în procesul de preprocesare a datelor.

            #### Cauze posibile ale outlierilor în setul nostru de date:

            1. **Erori de măsurare sau de înregistrare**: Unele valori extreme pot fi rezultatul unor erori umane sau tehnice.
            2. **Variabilitate naturală**: Unii pacienți pot avea într-adevăr valori mult diferite față de majoritatea populației.
            3. **Condiții medicale specifice**: Anumite condiții medicale pot cauza valori atipice pentru anumiți parametri.

            #### Impactul outlierilor:

            1. **Distorsionarea statisticilor descriptive**: Media și deviația standard sunt sensibile la outlieri.
            2. **Afectarea performanței modelelor**: Multe algoritmi de machine learning pot fi influențate negativ de prezența outlierilor.
            3. **Interpretări eronate**: Outlieri neidentificați pot duce la concluzii medicale incorecte.

            #### Opțiuni pentru tratarea outlierilor:

            1. **Păstrarea lor**: Dacă sunt valori legitime și reprezentative pentru anumite cazuri rare.
            2. **Eliminarea**: Dacă sunt erori clare sau cazuri care nu sunt de interes pentru analiza curentă.
            3. **Transformarea**: Aplicarea de transformări logaritmice sau alte metode pentru a reduce impactul valorilor extreme.
            4. **Winsorizing**: Înlocuirea valorilor extreme cu percentile specifice (de ex., percentilele 5 și 95).
            5. **Tratare separată**: Crearea de modele separate pentru cazurile atipice.

            Pentru acest set de date, recomandăm o abordare echilibrată, evaluând fiecare variabilă în parte și contextul medical specific, înainte de a decide strategia optimă pentru tratarea outlierilor.
            """
        )

        selected_column = st.selectbox(
            'Alegeți variabila pentru analiza outlierilor:',
            coloane_numerice
        )

        st.markdown(f"#### Analiza outlierilor pentru variabila: {selected_column}")

        Q1 = df_diabet[selected_column].quantile(0.25)
        Q3 = df_diabet[selected_column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df_diabet[(df_diabet[selected_column] < lower_bound) | (df_diabet[selected_column] > upper_bound)]

        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Limita inferioară:** {lower_bound:.2f}")
            st.write(f"**Limita superioară:** {upper_bound:.2f}")
            st.write(f"**Număr de outlieri:** {len(outliers)}")
            st.write(f"**Procent din total:** {(len(outliers) / len(df_diabet) * 100):.2f}%")

        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(y=df_diabet[selected_column], ax=ax, color="skyblue")
            ax.set_title(f'Boxplot pentru {selected_column}')
            st.pyplot(fig)

        if not outliers.empty:
            st.write("### Valorile outlier:")
            cols_to_show = [selected_column, 'Gender', 'Age', 'Diagnosis']
            cols_to_show.extend([col for col in ['BMI', 'FastingBloodSugar', 'HbA1c'] if col != selected_column])
            st.write(outliers[cols_to_show].head(20))

            fig, ax = plt.subplots(figsize=(12, 6))

            normal_values = df_diabet[~((df_diabet[selected_column] < lower_bound) |
                                        (df_diabet[selected_column] > upper_bound))][selected_column]

            ax.hist(normal_values, bins=30, alpha=0.7, label='Valori normale', color='blue')
            ax.hist(outliers[selected_column], bins=30, alpha=0.7, label='Outlieri', color='red')

            ax.set_title(f'Distribuția valorilor pentru {selected_column}')
            ax.set_xlabel(selected_column)
            ax.set_ylabel('Frecvență')
            ax.legend()

            st.pyplot(fig)
            st.success(f"Nu există outlieri pentru variabila {selected_column} conform metodei IQR.")

    # ---------------------------
    # Tab: Analiza distribuției datelor
    # ---------------------------
    with tab_analiza_distributie:
        st.markdown(
            "<h2 style='text-align: center; color: blue;'>Analiza distribuției datelor</h2>",
            unsafe_allow_html=True
        )

        vizualizare = st.radio(
            "Alegeți tipul de vizualizare:",
            ['Distribuții pentru toate variabilele numerice',
             'Comparație între pacienți cu/fără diabet',
             'Distribuții pentru variabile categorice']
        )

        if vizualizare == 'Distribuții pentru toate variabilele numerice':

            nr_grafice_rand = st.slider('Număr de grafice per rând:', min_value=1, max_value=4, value=2)

            nr_randuri = int(np.ceil(len(coloane_numerice) / nr_grafice_rand))
            fig, axes = plt.subplots(nrows=nr_randuri, ncols=nr_grafice_rand,
                                     figsize=(6 * nr_grafice_rand, 4 * nr_randuri))

            axes = axes.flatten()

            for i, col in enumerate(coloane_numerice):
                axes[i].hist(df_diabet[col], edgecolor='black', color='skyblue', bins=30)
                axes[i].set_title(f"Distribuție {col}")
                axes[i].set_xlabel(col)
                axes[i].set_ylabel("Frecvență")

            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)

            plt.tight_layout()
            st.pyplot(fig)

            st.markdown("""
            ### Interpretarea distribuțiilor numerice:

            Analizând distribuțiile variabilelor numerice, putem observa:

            1. **Variabile cu distribuție aproximativ normală (Gaussiană)**: Age, BMI, DietQuality, SleepQuality

            2. **Variabile cu distribuție asimetrică pozitivă (coadă spre dreapta)**: FastingBloodSugar, HbA1c, SerumCreatinine, CholesterolTriglycerides

            3. **Variabile cu distribuție bimodală**: Unele variabile ar putea prezenta două vârfuri, sugerând două subpopulații distincte în setul de date, posibil legate de prezența sau absența diabetului

            Aceste observații sunt importante pentru:
            - Alegerea metodelor adecvate de normalizare/standardizare
            - Identificarea necesității de transformare a datelor (ex: transformare logaritmică pentru distribuții asimetrice)
            - Detectarea potențialelor subpopulații care ar putea necesita analize separate
            """)

        elif vizualizare == 'Comparație între pacienți cu/fără diabet':

            selected_var = st.selectbox(
                'Alegeți variabila pentru comparare:',
                coloane_numerice
            )

            fig, ax = plt.subplots(figsize=(12, 6))

            ax.hist(df_neg_diabet[selected_var], bins=30, alpha=0.7,
                    label='Fără diabet', color='green')
            ax.hist(df_poz_diabet[selected_var], bins=30, alpha=0.7,
                    label='Cu diabet', color='red')

            ax.set_title(f'Distribuția {selected_var} în funcție de diagnostic')
            ax.set_xlabel(selected_var)
            ax.set_ylabel('Frecvență')
            ax.legend()

            st.pyplot(fig)

            fig2, ax2 = plt.subplots(figsize=(12, 6))
            sns.kdeplot(data=df_neg_diabet, x=selected_var, label='Fără diabet', color='green', ax=ax2)
            sns.kdeplot(data=df_poz_diabet, x=selected_var, label='Cu diabet', color='red', ax=ax2)
            ax2.set_title(f'Densitatea probabilității pentru {selected_var} în funcție de diagnostic')
            ax2.set_xlabel(selected_var)
            ax2.legend()

            st.pyplot(fig2)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"#### Statistici pentru pacienți fără diabet:")
                st.write(df_neg_diabet[selected_var].describe())

            with col2:
                st.markdown(f"#### Statistici pentru pacienți cu diabet:")
                st.write(df_poz_diabet[selected_var].describe())

            t_stat, p_val = scipy_stats.ttest_ind(
                df_neg_diabet[selected_var].dropna(),
                df_poz_diabet[selected_var].dropna(),
                equal_var=False
            )

            st.markdown(f"#### Testul t pentru diferența dintre medii:")
            st.write(f"Valoare t: {t_stat:.4f}")
            st.write(f"Valoare p: {p_val:.4f}")

            if p_val < 0.05:
                st.success(
                    f"Există o diferență semnificativă statistic în {selected_var} între pacienții cu și fără diabet (p < 0.05).")
            else:
                st.info(
                    f"Nu există o diferență semnificativă statistic în {selected_var} între pacienții cu și fără diabet (p > 0.05).")

        else:
            selected_cat = st.selectbox(
                'Alegeți variabila categorică:',
                ['Gender', 'Ethnicity', 'FamilyHistory', 'AntihypertensiveMedications', 'Statins',
                 'AntidiabeticMedications']
            )

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            sns.countplot(data=df_diabet, x=selected_cat, ax=ax1, palette='viridis')
            ax1.set_title(f'Distribuția pentru {selected_cat}')
            ax1.set_ylabel('Număr de pacienți')

            if selected_cat == 'Ethnicity':
                ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

            for p in ax1.patches:
                ax1.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='bottom')

            sns.countplot(data=df_diabet, x=selected_cat, hue='Diagnosis',
                          palette=['green', 'red'], ax=ax2)
            ax2.set_title(f'Distribuția pentru {selected_cat} în funcție de diagnostic')
            ax2.set_ylabel('Număr de pacienți')
            ax2.legend(title='Diagnostic', labels=['Fără diabet', 'Cu diabet'])

            if selected_cat == 'Ethnicity':
                ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

            plt.tight_layout()
            st.pyplot(fig)
            st.markdown("#### Tabel de contingență:")
            contingency_table = pd.crosstab(df_diabet[selected_cat], df_diabet['Diagnosis'])
            contingency_table.columns = ['Fără diabet', 'Cu diabet']
            st.write(contingency_table)

            percentage_table = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100
            percentage_table = percentage_table.round(2)

            st.markdown("#### Procentaje pe rânduri (%):")
            st.write(percentage_table)

            chi2, p, dof, expected = chi2_contingency(contingency_table)

            st.markdown("#### Test Chi-pătrat pentru independență:")
            st.write(f"Valoare Chi-pătrat: {chi2:.4f}")
            st.write(f"Valoare p: {p:.4f}")

            if p < 0.05:
                st.success(
                    f"Există o asociere semnificativă statistic între {selected_cat} și diagnosticul de diabet (p < 0.05).")
            else:
                st.info(
                    f"Nu există o asociere semnificativă statistic între {selected_cat} și diagnosticul de diabet (p > 0.05).")

            st.markdown("#### Interpretare:")

            if selected_cat == 'Gender':
                st.markdown("""
                            **Distribuția pe gen** arată proporția pacienților de gen masculin și feminin în setul de date și cum diferă riscul de diabet între aceste grupuri. Diferențele pot reflecta factori precum diferențele hormonale, obiceiurile de viață specifice fiecărui gen, și comportamentele de căutare a asistenței medicale.
                            """)
            elif selected_cat == 'Ethnicity':
                st.markdown("""
                            **Distribuția etnică** este importantă deoarece anumite grupuri etnice au un risc genetic mai mare pentru diabet tip 2. Această informație poate ajuta la identificarea grupurilor care ar putea beneficia de screening mai intensiv sau intervenții preventive personalizate.
                            """)
            elif selected_cat == 'FamilyHistory':
                st.markdown("""
                            **Istoricul familial** este un factor de risc cunoscut pentru diabetul tip 2. Un test chi-pătrat semnificativ confirmă importanța predispoziției genetice și a factorilor de mediu familiali în dezvoltarea diabetului.
                            """)
            elif selected_cat in ['AntihypertensiveMedications', 'Statins', 'AntidiabeticMedications']:
                st.markdown(f"""
                            **Utilizarea de {selected_cat}** poate reflecta atât comorbidități (precum hipertensiunea sau dislipidemia) cât și abordări terapeutice. Asocierea semnificativă cu diagnosticul de diabet indică suprapunerea între aceste afecțiuni și diabetul zaharat, subliniind importanța unei abordări integrate în managementul acestor condiții.
                            """)

            # ---------------------------
            # Tab: Grafic pairplot
            # ---------------------------
        with tab_grafic_pairplot:
            st.markdown(
                "<h2 style='text-align: center; color: blue;'>Analiza graficului pairplot</h2>",
                unsafe_allow_html=True
            )

            st.warning("""
                    Generarea unui pairplot complet pentru toate variabilele numerice poate dura mult timp și poate consuma multe resurse.
                    Vă recomandăm să selectați un subset de variabile de interes.
                    """)

            selected_vars = st.multiselect(
                'Selectați variabilele pentru pairplot (maximum 6 recomandat):',
                coloane_numerice,
                default=['Age', 'BMI', 'FastingBloodSugar', 'HbA1c']
            )

            if len(selected_vars) < 2:
                st.error("Vă rugăm să selectați cel puțin 2 variabile pentru a genera pairplot-ul.")
            else:
                if len(selected_vars) > 6:
                    st.warning("Ați selectat multe variabile, generarea pairplot-ului poate dura mai mult.")

                plot_data = df_diabet[selected_vars + ['Diagnosis']].copy()
                plot_data['Diagnostic'] = plot_data['Diagnosis'].map({0: 'Fără diabet', 1: 'Cu diabet'})

                fig = plt.figure(figsize=(12, 10))
                g = sns.pairplot(plot_data, hue='Diagnostic',
                                 palette={'Fără diabet': 'green', 'Cu diabet': 'red'},
                                 diag_kind="kde")
                g.fig.suptitle("Pairplot pentru variabilele selectate", y=1.02, fontsize=16)
                st.pyplot(g.fig)

                st.markdown("""
                        ### Interpretarea pairplot-ului:

                        Pairplot-ul este un instrument vizual puternic care permite:

                        1. **Vizualizarea distribuțiilor** individuale ale variabilelor (pe diagonală)
                        2. **Examinarea relațiilor bivariaționate** între toate perechile de variabile
                        3. **Identificarea clusterelor** și a separabilității între pacienții cu și fără diabet

                        #### Ce putem observa:

                        - **Corelații pozitive** apar ca tendințe crescătoare de la stânga jos la dreapta sus
                        - **Corelații negative** apar ca tendințe descrescătoare de la stânga sus la dreapta jos
                        - **Diferențe în distribuții** între grupul cu diabet (roșu) și fără diabet (verde)
                        - **Potențiale granițe de decizie** care ar putea separa cele două grupuri

                        Acest tip de vizualizare este deosebit de util pentru identificarea variabilelor care contribuie cel mai mult la separarea pacienților cu și fără diabet, oferind perspective valoroase pentru modelarea predictivă ulterioară.
                        """)

            # ---------------------------
            # Tab: Analiza corelațiilor
            # ---------------------------
        with tab_analiza_corelatii:
            st.markdown(
                "<h2 style='text-align: center; color: blue;'>Analiza corelațiilor dintre variabile</h2>",
                unsafe_allow_html=True
            )

            correlation_type = st.radio(
                "Alegeți tipul de corelație:",
                ['Pearson (parametrică)', 'Spearman (non-parametrică)']
            )

            if correlation_type == 'Pearson (parametrică)':
                matrice_corelatie = df_diabet[coloane_numerice + ['Diagnosis']].corr(method='pearson')
                corr_method = "Pearson"
            else:
                matrice_corelatie = df_diabet[coloane_numerice + ['Diagnosis']].corr(method='spearman')
                corr_method = "Spearman"
            show_option = st.radio(
                "Vizualizare:",
                ['Toate corelațiile', 'Doar corelații cu diagnosticul']
            )

            if show_option == 'Doar corelații cu diagnosticul':
                diag_corr = matrice_corelatie['Diagnosis'].drop('Diagnosis').sort_values(ascending=False)

                diag_corr_df = pd.DataFrame({
                    'Variabilă': diag_corr.index,
                    f'Corelație {corr_method} cu diagnosticul': diag_corr.values
                })

                st.markdown(f"#### Corelații {corr_method} cu diagnosticul de diabet (sortate):")
                st.write(diag_corr_df)

                fig, ax = plt.subplots(figsize=(12, 10))
                bars = ax.barh(diag_corr.index, diag_corr.values, color=plt.cm.RdBu_r(0.5 + diag_corr.values / 2))
                ax.set_xlabel(f'Corelație {corr_method}')
                ax.set_title(f'Corelații {corr_method} cu diagnosticul de diabet')
                ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)

                for i, v in enumerate(diag_corr.values):
                    ax.text(v + 0.01 if v >= 0 else v - 0.07, i, f"{v:.3f}", va='center')

                st.pyplot(fig)


            else:
                fig, ax = plt.subplots(figsize=(14, 12))
                mask = np.triu(np.ones_like(matrice_corelatie, dtype=bool))
                sns.heatmap(matrice_corelatie, annot=True, fmt=".2f", cmap="coolwarm",
                            mask=mask, vmin=-1, vmax=1, ax=ax, cbar_kws={"shrink": .8})
                ax.set_title(f"Matricea de corelație {corr_method} pentru variabilele numerice")
                plt.tight_layout()
                st.pyplot(fig)

            st.markdown(
                """
                ### Interpretarea matricei de corelații:

                Analizând matricea de corelații pentru variabilele numerice, putem observa următoarele relații semnificative:

                - Între FastingBloodSugar și HbA1c: Acest lucru este de așteptat din punct de vedere medical, deoarece HbA1c reprezintă nivelul mediu al glicemiei pe o perioadă de 2-3 luni.
                - Între SystolicBP și DiastolicBP: Aceste două măsurători ale tensiunii arteriale tind să varieze împreună.
                - Între CholesterolTotal și CholesterolLDL: Colesterolul total include LDL, deci această corelație este logică.

                - Între BMI și factori de risc metabolic: Un BMI ridicat tinde să fie asociat cu niveluri mai ridicate de colesterol și glicemie.
                - Între SleepQuality și PhysicalActivity: Persoanele care sunt mai active fizic tind să aibă un somn de calitate mai bună.
                - Între DietQuality și HealthLiteracy: Persoanele cu o educație medicală mai bună tind să aibă o dietă mai sănătoasă.

                - Între PhysicalActivity și FatigueLevels: Mai multă activitate fizică tinde să fie asociată cu niveluri mai scăzute de oboseală.
                - Între DietQuality și FastingBloodSugar: O dietă mai bună este asociată cu niveluri mai scăzute ale glicemiei.

                - FastingBloodSugar și HbA1c: Corelații foarte puternice, fiind criterii diagnostice pentru diabet.
                - BMI, Age și SystolicBP: Corelații moderate, reflectând factorii de risc cunoscuți pentru diabet.
                - PhysicalActivity și DietQuality: Corelații negative, indicând rolul protector al unui stil de viață sănătos.

                Aceste informații despre corelații ne ajută să înțelegem relațiile dintre variabile și ne pot ghida în selectarea caracteristicilor pentru modelele predictive.
                """
            )

            # ---------------------------
            # Tab: Encodarea variabilelor discrete
            # ---------------------------
        with tab_encodare:
            st.markdown(
                "<h2 style='text-align: center; color: blue;'>Encodarea variabilelor categorice</h2>",
                unsafe_allow_html=True
            )

            st.markdown(
                """
                ## Encodarea variabilelor categorice

                Variabilele categorice din setul nostru de date care necesită encodare:

                - **Ethnicity**: Reprezintă originea etnică a pacienților și nu are o ordine naturală între categorii.
                - **Gender**: Variabilă binară care poate fi encodată simplu (0/1).
                - **Occupation**: Reprezintă ocupația pacienților și nu are o ordine naturală.
                - **FamilyHistory**: Istoricul familial de diabet, variabilă binară.
                - **Medications**: Medicamentele utilizate de pacienți, care ar putea fi encodate individual.

                Pentru aceste variabile, vom aplica:

                1. **Label Encoding** pentru variabilele binare (Gender, FamilyHistory)
                2. **One-Hot Encoding** pentru variabilele categorice nominale (Ethnicity, Occupation)

                Acest proces este esențial pentru a transforma datele categorice în format numeric pentru utilizare în modelele de machine learning, păstrând în același timp informația categorială fără a introduce relații ordinale artificiale.
                """
            )

            st.markdown("### One-Hot Encoding pentru variabile nominale")
            st.markdown("""
                    **Avantajele Label Encoding:**
                    - Simplu și eficient
                    - Păstrează relațiile ordinale (dacă există)
                    - Produce o singură coloană nouă

                    **Dezavantajele Label Encoding:**
                    - Poate introduce ordine artificială între categorii
                    - Poate fi problematic pentru algoritmi care presupun relații ordinale între valori

                    #### Setul nostru de date nu are variabile care au valori literale, deci nu este nevoie de lable encoding.""")

            st.markdown("### One-Hot Encoding pentru variabile nominale")

            nominal_var = st.selectbox(
                'Selectați o variabilă nominală pentru One-Hot Encoding:',
                ['Ethnicity']
            )

            dummies = pd.get_dummies(df_diabet[nominal_var], prefix=nominal_var)

            st.markdown("#### Exemplu de One-Hot Encoding (primele 10 rânduri):")
            display_df = pd.concat([df_diabet[nominal_var].head(10).reset_index(drop=True),
                                    dummies.head(10).reset_index(drop=True)], axis=1)
            st.write(display_df)

            st.markdown("""
                    **Avantajele One-Hot Encoding:**
                    - Nu introduce relații ordinale artificiale
                    - Adecvat pentru variabile nominale
                    - Funcționează bine cu majoritatea algoritmilor de machine learning

                    **Dezavantajele One-Hot Encoding:**
                    - Creează multe coloane noi pentru variabile cu multe categorii
                    - Poate duce la dimensionalitate ridicată
                    - Poate introduce multicolinearitate (dacă nu se elimină una dintre coloane)
                    """)

            st.markdown("### Aplicarea encodării pe întregul set de date")

            if st.button("Aplică encodarea pe întregul set de date"):
                df_encoded = df_diabet.copy()

                nominal_vars = ['Ethnicity']
                df_encoded = pd.get_dummies(df_encoded, columns=nominal_vars, drop_first=True)

                st.success("Encodarea a fost aplicată cu succes!")
                st.write(df_encoded.head())

                st.markdown("#### Structura noului set de date:")
                st.write(f"Număr de coloane înainte de encodare: {df_diabet.shape[1]}")
                st.write(f"Număr de coloane după encodare: {df_encoded.shape[1]}")
                st.success(
                    "A fost eliminata prima coloana si anume Ethnicity_0 pentru a se evita coliniaritatea problema „dummy variable trap”")

                new_cols = [col for col in df_encoded.columns if col not in df_diabet.columns]
                st.markdown("#### Noile coloane adăugate prin encodare:")
                st.write(new_cols)

            # ---------------------------
            # Tab: Standardizarea datelor
            # ---------------------------
        with tab_standardizare:
            st.markdown(
                "<h2 style='text-align: center; color: blue;'>Standardizarea datelor numerice</h2>",
                unsafe_allow_html=True
            )

            st.markdown(
                """
                ## Standardizarea setului de date

                Standardizarea (sau Z-score normalization) este procesul prin care variabilele numerice sunt transformate pentru a avea o medie de 0 și o deviație standard de 1. Formula utilizată este:

                $z = \\frac{x - \\mu}{\\sigma}$

                unde:
                - $x$ este valoarea originală
                - $\\mu$ este media variabilei
                - $\\sigma$ este deviația standard a variabilei

                ### Avantajele standardizării:

                1. **Uniformizarea scalelor**: Variabile care inițial au fost măsurate în unități diferite (vârstă, BMI, etc.) sunt aduse la o scală comună.
                2. **Îmbunătățirea performanței algoritmilor**: Multe algoritme de machine learning (în special cele bazate pe distanță sau gradient) funcționează mai bine când datele sunt standardizate.
                3. **Reducerea influenței valorilor extreme**: Valorile extreme sunt aduse mai aproape de medie, deși standardizarea nu le elimină complet.
                """
            )

            selected_vars_std = st.multiselect(
                'Selectați variabilele pentru standardizare:',
                coloane_numerice,
                default=['Age', 'BMI', 'FastingBloodSugar', 'HbA1c']
            )

            if not selected_vars_std:
                st.warning("Vă rugăm să selectați cel puțin o variabilă pentru standardizare.")
            else:
                df_std = df_diabet.copy()
                scaler = StandardScaler()
                df_std[selected_vars_std] = scaler.fit_transform(df_std[selected_vars_std])

                st.markdown("### Statistici înainte și după standardizare:")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Înainte de standardizare:")
                    st.write(df_diabet[selected_vars_std].describe().loc[["mean", "std"]])

                with col2:
                    st.markdown("#### După standardizare:")
                    st.write(df_std[selected_vars_std].describe().loc[["mean", "std"]])

                st.markdown("### Comparație distribuții înainte și după standardizare:")

                std_var_to_viz = st.selectbox(
                    'Selectați variabila pentru vizualizare:',
                    selected_vars_std,
                    key='std_viz'
                )

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

                sns.histplot(df_diabet[std_var_to_viz], kde=True, ax=ax1, color='blue')
                ax1.set_title(f'Distribuția originală - {std_var_to_viz}')
                ax1.set_xlabel(std_var_to_viz)

                sns.histplot(df_std[std_var_to_viz], kde=True, ax=ax2, color='green')
                ax2.set_title(f'Distribuția standardizată - {std_var_to_viz}')
                ax2.set_xlabel(f'{std_var_to_viz} (standardizat)')

                plt.tight_layout()
                st.pyplot(fig)

                st.markdown("### Exemplu de date înainte și după standardizare (primele 10 rânduri):")

                example_df = pd.DataFrame({
                    f'{std_var_to_viz} (original)': df_diabet[std_var_to_viz].head(10).values,
                    f'{std_var_to_viz} (standardizat)': df_std[std_var_to_viz].head(10).values
                })

                st.write(example_df)

                st.markdown("""
                        ### Observații după standardizare:

                        După cum se poate observa în tabelul de mai sus, toate variabilele standardizate au acum o medie apropiată de 0 și o deviație standard apropiată de 1. Acest lucru confirmă aplicarea corectă a standardizării.

                        ### Când să folosim standardizarea:

                        Standardizarea este recomandată în special pentru:

                        1. **Algoritmi bazați pe distanță**: K-Nearest Neighbors, K-Means, SVM
                        2. **Algoritmi bazați pe gradient**: Regresia logistică, rețele neuronale
                        3. **Metode care implică reducerea dimensionalității**: PCA, t-SNE
                        4. **Date cu scale diferite**: Când variabilele au unități de măsură diferite
                        5. **Când vrem să comparăm coeficienți**: În regresie, pentru a compara importanța relativă a variabilelor

                        Nu este necesară pentru algoritmi bazați pe arbori de decizie (Decision Trees, Random Forest, Gradient Boosting), care sunt invarianți la transformări monotone.
                        """)

            # ---------------------------
            # Tab: Normalizarea datelor
            # ---------------------------
        with tab_normalizare:
            st.markdown(
                "<h2 style='text-align: center; color: blue;'>Normalizarea datelor numerice</h2>",
                unsafe_allow_html=True
            )

            st.markdown(
                """
                ## Normalizarea setului de date

                Normalizarea (sau Min-Max scaling) este procesul prin care variabilele numerice sunt transformate pentru a fi în intervalul [0, 1]. Formula utilizată este:

                $x_{scaled} = \\frac{x - x_{min}}{x_{max} - x_{min}}$

                unde:
                - $x$ este valoarea originală
                - $x_{min}$ este valoarea minimă a variabilei
                - $x_{max}$ este valoarea maximă a variabilei

                ### Avantajele normalizării:

                1. **Interval fix**: Toate valorile sunt aduse în intervalul [0, 1], ceea ce face interpretarea ușoară.
                2. **Păstrarea distribuției**: Spre deosebire de standardizare, normalizarea păstrează forma distribuției originale.
                3. **Utilitate pentru algoritmi care necesită valori pozitive**: Unii algoritmi, precum Naive Bayes, beneficiază de date normalizate.
                """
            )

            selected_vars_norm = st.multiselect(
                'Selectați variabilele pentru normalizare:',
                coloane_numerice,
                default=['Age', 'BMI', 'FastingBloodSugar', 'HbA1c'],
                key='norm_vars'
            )

            if not selected_vars_norm:
                st.warning("Vă rugăm să selectați cel puțin o variabilă pentru normalizare.")
            else:
                df_norm = df_diabet.copy()
                minmax_scaler = MinMaxScaler()
                df_norm[selected_vars_norm] = minmax_scaler.fit_transform(df_norm[selected_vars_norm])

                st.markdown("### Statistici înainte și după normalizare:")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Înainte de normalizare:")
                    st.write(df_diabet[selected_vars_norm].describe().loc[["min", "max"]])

                with col2:
                    st.markdown("#### După normalizare:")
                    st.write(df_norm[selected_vars_norm].describe().loc[["min", "max"]])

                st.markdown("### Comparație distribuții înainte și după normalizare:")

                norm_var_to_viz = st.selectbox(
                    'Selectați variabila pentru vizualizare:',
                    selected_vars_norm,
                    key='norm_viz'
                )

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

                sns.histplot(df_diabet[norm_var_to_viz], kde=True, ax=ax1, color='blue')
                ax1.set_title(f'Distribuția originală - {norm_var_to_viz}')
                ax1.set_xlabel(norm_var_to_viz)

            sns.histplot(df_norm[norm_var_to_viz], kde=True, ax=ax2, color='orange')
            ax2.set_title(f'Distribuția normalizată - {norm_var_to_viz}')
            ax2.set_xlabel(f'{norm_var_to_viz} (normalizat)')

            plt.tight_layout()
            st.pyplot(fig)

            st.markdown("### Exemplu de date înainte și după normalizare (primele 10 rânduri):")

            example_df = pd.DataFrame({
                f'{norm_var_to_viz} (original)': df_diabet[norm_var_to_viz].head(10).values,
                f'{norm_var_to_viz} (normalizat)': df_norm[norm_var_to_viz].head(10).values
            })

            st.write(example_df)

            st.markdown("### Comparație între diferite metode de scalare:")

            if st.checkbox("Arată comparația între standardizare și normalizare"):
                var_compare = st.selectbox(
                    'Selectați variabila pentru comparație:',
                    selected_vars_norm,
                    key='compare_scales'
                )

                std_values = StandardScaler().fit_transform(df_diabet[[var_compare]]).flatten()

                norm_values = MinMaxScaler().fit_transform(df_diabet[[var_compare]]).flatten()

                robust_values = RobustScaler().fit_transform(df_diabet[[var_compare]]).flatten()

                compare_df = pd.DataFrame({
                    'Original': df_diabet[var_compare].values,
                    'Standardizare (z-score)': std_values,
                    'Normalizare (min-max)': norm_values,
                    'Scalare robustă': robust_values
                })

                st.write(compare_df.head(10))

                fig, axs = plt.subplots(4, 1, figsize=(10, 16))

                sns.histplot(compare_df['Original'], kde=True, ax=axs[0], color='blue')
                axs[0].set_title(f'Distribuția originală - {var_compare}')

                sns.histplot(compare_df['Standardizare (z-score)'], kde=True, ax=axs[1], color='green')
                axs[1].set_title(f'Standardizare (z-score) - {var_compare}')

                sns.histplot(compare_df['Normalizare (min-max)'], kde=True, ax=axs[2], color='orange')
                axs[2].set_title(f'Normalizare (min-max) - {var_compare}')

                sns.histplot(compare_df['Scalare robustă'], kde=True, ax=axs[3], color='purple')
                axs[3].set_title(f'Scalare robustă - {var_compare}')

                plt.tight_layout()
                st.pyplot(fig)

                st.markdown("#### Statistici comparative:")
                st.write(compare_df.describe())

            st.markdown("""
            ### Observații după normalizare:

            După cum se poate observa în tabelul de mai sus, toate variabilele normalizate au acum valori minime aproape de 0 și valori maxime aproape de 1. Aceasta confirmă aplicarea corectă a normalizării.

            ### Când să folosim normalizarea:

            Normalizarea este recomandată când:
            - Știm că distribuția nu este Gaussiană (normală)
            - Avem nevoie de valori pozitive în intervalul [0, 1]
            - Folosim algoritmi precum K-Nearest Neighbors sau rețele neuronale cu funcții de activare sigmoidale
            - Lucrăm cu algoritmi de procesare a imaginilor sau rețele neuronale convoluționale
            - Avem de-a face cu date ce conțin caracteristici cu scale diferite dar fără outlieri semnificativi

            ### Diferențe principale între standardizare și normalizare:

            | Aspect | Standardizare (Z-score) | Normalizare (Min-Max) |
            |--------|--------------------------|------------------------|
            | Interval rezultat | Fără limite fixe | [0, 1] |
            | Sensibilitate la outlieri | Moderată | Mare |
            | Distribuția rezultată | Centrat la 0, SD = 1 | Păstrează forma distribuției |
            | Utilizare ideală | Date aproximativ normale | Date non-gaussiene |
            | Algoritmi recomandați | SVM, Regresie logistică | KNN, Rețele neuronale |

            ### Scalarea robustă:

            O alternativă utilă atât la standardizare cât și la normalizare este **scalarea robustă**, care utilizează statistici robuste (mediană și IQR) în loc de medie și deviație standard. Aceasta este mai puțin sensibilă la prezența outlierilor și poate fi o alegere mai bună pentru seturi de date cu valori extreme.
            """)

st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7f9;
    }
    h1, h2, h3, h4 {
        color: #1E3A8A;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #E0E7FF;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)
