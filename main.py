import pandas as pd
import seaborn as sns
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
)
from scipy import stats as scipy_stats
from scipy.stats import chi2_contingency

from cluster import page_analiza_clustering_diabet
from regresie import page_model_regresie_diabet, page_model_regresie_statsmodels

st.set_page_config(
    page_title="Analiza datelor despre diabet", page_icon="📊", layout="wide"
)

if "df_std" not in st.session_state:
    st.session_state.df_std = None

@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Încarca datele din fișier CSV și le returneaza ca DataFrame.
    """
    df = pd.read_csv("./data/diabetes_data.csv")
    return df


# -------------------------------
# Funcții ajutatoare pentru curațare
# -------------------------------

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina liniile duplicate din DataFrame.
    Afișeaza mesaj daca s-au eliminat duplicate.
    """
    initial_count = df.shape[0]
    df_no_dup = df.drop_duplicates()
    final_count = df_no_dup.shape[0]

    if initial_count == final_count:
        st.success("Setul de date nu conține înregistrari duplicate!")
    else:
        st.warning(
            f"Au fost eliminate {initial_count - final_count} înregistrari duplicate. "
            f"Acum setul conține {final_count} înregistrari."
        )

    return df_no_dup


def summarize_missing_values(df: pd.DataFrame):
    """
    Verifica și afișeaza contorul valorilor lipsa pe coloane.
    """
    nan_summary = df.isnull().sum()
    total_missing = nan_summary.sum()
    if total_missing > 0:
        st.warning("Setul de date conține valori lipsa:")
        st.write(nan_summary[nan_summary > 0])
    else:
        st.success("Setul de date nu conține valori lipsa!")


def drop_irrelevant_columns(df: pd.DataFrame, cols_to_drop: list[str]) -> pd.DataFrame:
    """
    Șterge coloanele care nu sunt relevante pentru analiza.
    """
    return df.drop(cols_to_drop, axis=1, errors="ignore")


# -------------------------------
# Funcții pentru secțiunea "Detalii despre setul de date"
# -------------------------------

def show_dataset_details(df: pd.DataFrame):
    """
    Afișeaza tabelul complet și descrierea variabilelor.
    Elimina duplicatele și afișeaza valorile lipsa.
    """
    st.write(df)

    st.subheader("Variabilele setului de date:")
    try:
        with open("explicare_variabile.txt", "r") as file:
            continut = file.read()
        st.markdown(continut)
    except FileNotFoundError:
        st.markdown(
            """
        ## Descrierea variabilelor din setul de date:clean_initial_data
        - **PatientID**: ID-ul unic al pacientului  
        - **Gender**: Genul pacientului  
        - **Age**: Vârsta pacientului  
        - **Ethnicity**: Originea etnica a pacientului  
        - **Occupation**: Ocupația pacientului  
        - **BMI**: Indicele de masa corporala  
        - **AlcoholConsumption**: Nivelul de consum de alcool  
        - **PhysicalActivity**: Nivel de activitate fizica (minute/saptamâna)  
        - **DietQuality**: Calitatea dietei (1–10)  
        - **SleepQuality**: Calitatea somnului (1–10)  
        - **SystolicBP**: Tensiunea arteriala sistolica  
        - **DiastolicBP**: Tensiunea arteriala diastolica  
        - **FastingBloodSugar**: Glicemia à jeun (mg/dL)  
        - **HbA1c**: Hemoglobina glicata (%)  
        - **SerumCreatinine**: Creatinina serica (mg/dL)  
        - **BUNLevels**: Uree sanguina (mg/dL)  
        - **CholesterolTotal**: Colesterol total (mg/dL)  
        - **CholesterolLDL**: Colesterol LDL (mg/dL)  
        - **CholesterolHDL**: Colesterol HDL (mg/dL)  
        - **CholesterolTriglycerides**: Trigliceride (mg/dL)  
        - **FatigueLevels**: Nivel de oboseala (1–10)  
        - **MedicalCheckupsFrequency**: Frecvența controalelor medicale/an  
        - **MedicationAdherence**: Aderența la medicație (1–10)  
        - **HealthLiteracy**: Cunoștințe medicale (1–10)  
        - **FamilyHistory**: Istoric familial de diabet (0 = Nu, 1 = Da)  
        - **AntihypertensiveMedications**: Utilizare medicamente antihipertensive (0 = Nu, 1 = Da)  
        - **Statins**: Utilizare statine (0 = Nu, 1 = Da)  
        - **AntidiabeticMedications**: Utilizare medicamente antidiabetice (0 = Nu, 1 = Da)  
        - **DoctorInCharge**: ID-ul medicului responsabil  
        - **Diagnosis**: Diagnosticul de diabet (0 = Nu, 1 = Da)  
        """
        )

    df_clean = remove_duplicates(df)
    summarize_missing_values(df_clean)


# -------------------------------
# Funcții pentru Analiza Exploratorie
# -------------------------------

def clean_initial_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina coloanele irelevante (PatientID, DoctorInCharge) și returneaza DataFrame-ul curațat.
    """
    cols_to_drop = ["PatientID", "DoctorInCharge"]
    df_clean = drop_irrelevant_columns(df, cols_to_drop)
    return df_clean


def descriptive_general(df_clean: pd.DataFrame, coloane_numerice: list[str]):
    """
    Afișeaza statistici descriptive pentru pacienți cu și fara diabet,
    plus un barplot al distribuției coloanei 'Diagnosis'.
    """
    df_poz = df_clean[df_clean["Diagnosis"] == 1]
    df_neg = df_clean[df_clean["Diagnosis"] == 0]

    st.markdown(
        "<h2 style='text-align: center; color: blue;'>Analiza generala a datelor numerice</h2>",
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "<h3 style='text-align: center; color: red;'>Pacienți cu diabet</h3>",
            unsafe_allow_html=True,
        )
        st.write(df_poz[coloane_numerice].describe())

    with col2:
        st.markdown(
            "<h3 style='text-align: center; color: green;'>Pacienți fara diabet</h3>",
            unsafe_allow_html=True,
        )
        st.write(df_neg[coloane_numerice].describe())

    st.markdown(
        "<h2 style='text-align: center; color: red;'>Interpretarea setului de date</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        Acest set de date conține **{len(df_clean)} observații** și masoara diverse variabile clinice.

        **Distribuția diagnosticului:** {df_clean['Diagnosis'].value_counts()[1]} pacienți cu diabet și
        {df_clean['Diagnosis'].value_counts()[0]} pacienți fara diabet.
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### Observații descriptive:
        - Pacienții cu diabet au medii mai mari la vârsta, BMI, glicemie à jeun, HbA1c, tensiune arteriala,
          colesterol total/LDL/trigliceride.
        - Pacienții fara diabet au niveluri mai ridicate de activitate fizica, dieta și somn.

        În ansamblu, sunt confirmați factorii de risc: vârsta crescuta, exces ponderal, sedentarism,
        alimentație inadecvata, comorbiditați cardiovasculare și dislipidemie.
        """,
        unsafe_allow_html=True,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x="Diagnosis", data=df_clean, palette=["green", "red"], ax=ax)
    ax.set_title("Distribuția diagnosticului de diabet")
    ax.set_xlabel("Diagnostic (0 = Nu, 1 = Da)")
    ax.set_ylabel("Numar de pacienți")
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height()}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
        )
    st.pyplot(fig)


def analyze_outliers(df_clean: pd.DataFrame, coloane_numerice: list[str]):
    """
    Realizeaza analiza outlier-ilor pentru coloana selectata.
    """
    st.markdown(
        "<h2 style='text-align: center; color: red;'>Analiza valorilor extreme (outlier)</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        Outlier-ii pot distorsiona statistici și modele ML. 
        Metode: Elimination, transformare (log), winsorizing etc.
        """
    )

    selected_column = st.selectbox(
        "Alege variabila pentru analiza outlierilor:", coloane_numerice
    )
    st.markdown(f"#### Analiza outlierilor pentru: **{selected_column}**")

    Q1 = df_clean[selected_column].quantile(0.25)
    Q3 = df_clean[selected_column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df_clean[
        (df_clean[selected_column] < lower_bound)
        | (df_clean[selected_column] > upper_bound)
    ]

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Limita inferioara:** {lower_bound:.2f}")
        st.write(f"**Limita superioara:** {upper_bound:.2f}")
        st.write(f"**Numar outlieri:** {len(outliers)}")
        st.write(f"**Procent din total:** {(len(outliers) / len(df_clean)*100):.2f}%")

    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(y=df_clean[selected_column], ax=ax, color="skyblue")
        ax.set_title(f"Boxplot pentru {selected_column}")
        st.pyplot(fig)

    if not outliers.empty:
        st.write("### Exemple outlieri (primele 20):")
        cols_to_show = [selected_column, "Gender", "Age", "Diagnosis"]
        extras = [col for col in ["BMI", "FastingBloodSugar", "HbA1c"] if col != selected_column]
        cols_to_show.extend(extras)
        st.write(outliers[cols_to_show].head(20))

        fig, ax = plt.subplots(figsize=(12, 6))
        normal_values = df_clean[
            (df_clean[selected_column] >= lower_bound)
            & (df_clean[selected_column] <= upper_bound)
        ][selected_column]
        ax.hist(normal_values, bins=30, alpha=0.7, label="Valori normale", color="blue")
        ax.hist(outliers[selected_column], bins=30, alpha=0.7, label="Outlieri", color="red")
        ax.set_title(f"Distribuția {selected_column}")
        ax.set_xlabel(selected_column)
        ax.set_ylabel("Frecvența")
        ax.legend()
        st.pyplot(fig)


def plot_numeric_distributions(df_clean: pd.DataFrame, coloane_numerice: list[str]):
    """
    Afișeaza histograme pentru toate variabilele numerice sau comparații între grupuri.
    """
    st.markdown(
        "<h2 style='text-align: center; color: blue;'>Analiza distribuției datelor</h2>",
        unsafe_allow_html=True,
    )

    df_poz = df_clean[df_clean["Diagnosis"] == 1]
    df_neg = df_clean[df_clean["Diagnosis"] == 0]

    vizualizare = st.radio(
        "Alege tipul de vizualizare:",
        [
            "Distribuții toate variabilele numerice",
            "Comparație pacienți cu/fara diabet",
            "Distribuții variabile categorice",
        ],
    )

    if vizualizare == "Distribuții toate variabilele numerice":
        nr_per_row = st.slider(
            "Numar grafice per rând:", min_value=1, max_value=4, value=2
        )
        nr_rows = int(np.ceil(len(coloane_numerice) / nr_per_row))
        fig, axes = plt.subplots(
            nrows=nr_rows,
            ncols=nr_per_row,
            figsize=(6 * nr_per_row, 4 * nr_rows),
        )
        axes = axes.flatten()
        for idx, col in enumerate(coloane_numerice):
            axes[idx].hist(df_clean[col], bins=30, edgecolor="black", color="skyblue")
            axes[idx].set_title(f"Distribuție {col}")
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel("Frecvența")
        for j in range(idx + 1, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown(
            """
        ### Observații:
        - Variabile aproximativ normale: Age, BMI, DietQuality, SleepQuality  
        - Variabile asimetrice pozitive: FastingBloodSugar, HbA1c, SerumCreatinine, CholesterolTriglycerides  
        - Posibile distribuții bimodale (subpopulații: cu/fara diabet)
        """
        )

    elif vizualizare == "Comparație pacienți cu/fara diabet":
        selected_var = st.selectbox("Alege variabila pentru comparație:", coloane_numerice)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(
            df_neg[selected_var], bins=30, alpha=0.7, label="Fara diabet", color="green"
        )
        ax.hist(
            df_poz[selected_var], bins=30, alpha=0.7, label="Cu diabet", color="red"
        )
        ax.set_title(f"Distribuția {selected_var} în funcție de diagnosticul de diabet")
        ax.set_xlabel(selected_var)
        ax.set_ylabel("Frecvența")
        ax.legend()
        st.pyplot(fig)

        fig2, ax2 = plt.subplots(figsize=(12, 6))
        sns.kdeplot(data=df_neg, x=selected_var, label="Fara diabet", color="green", ax=ax2)
        sns.kdeplot(data=df_poz, x=selected_var, label="Cu diabet", color="red", ax=ax2)
        ax2.set_title(f"Densitatea {selected_var} în funcție de diagnosticul de diabet")
        ax2.set_xlabel(selected_var)
        ax2.legend()
        st.pyplot(fig2)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Statistici (fara diabet):")
            st.write(df_neg[selected_var].describe())
        with col2:
            st.markdown("#### Statistici (cu diabet):")
            st.write(df_poz[selected_var].describe())

        t_stat, p_val = scipy_stats.ttest_ind(
            df_neg[selected_var].dropna(),
            df_poz[selected_var].dropna(),
            equal_var=False,
        )
        st.markdown("#### Test t de diferența a mediilor:")
        st.write(f"Valoare t = {t_stat:.4f}, p = {p_val:.4f}")
        if p_val < 0.05:
            st.success(
                f"Exista o diferența semnificativa în {selected_var} (p < 0.05)."
            )
        else:
            st.info(
                f"Nu exista diferența semnificativa în {selected_var} (p ≥ 0.05)."
            )

    else:  # "Distribuții variabile categorice"
        selected_cat = st.selectbox(
            "Alege variabila categorica:",
            [
                "Gender",
                "Ethnicity",
                "FamilyHistory",
                "AntihypertensiveMedications",
                "Statins",
                "AntidiabeticMedications",
            ],
        )
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        sns.countplot(data=df_clean, x=selected_cat, ax=ax1, palette="viridis")
        ax1.set_title(f"Distribuția {selected_cat}")
        ax1.set_ylabel("Numar de pacienți")
        if selected_cat == "Ethnicity":
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")
        for p in ax1.patches:
            ax1.annotate(
                f"{p.get_height()}",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="bottom",
            )

        sns.countplot(
            data=df_clean,
            x=selected_cat,
            hue="Diagnosis",
            palette=["green", "red"],
            ax=ax2,
        )
        ax2.set_title(f"{selected_cat} vs. Diagnostic")
        ax2.set_ylabel("Numar de pacienți")
        ax2.legend(title="Diagnostic", labels=["Fara diabet", "Cu diabet"])
        if selected_cat == "Ethnicity":
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")

        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("#### Tabel de contingența:")
        cont_table = pd.crosstab(df_clean[selected_cat], df_clean["Diagnosis"])
        cont_table.columns = ["Fara diabet", "Cu diabet"]
        st.write(cont_table)

        pct_table = cont_table.div(cont_table.sum(axis=1), axis=0) * 100
        st.markdown("#### Procente pe rânduri (%):")
        st.write(pct_table.round(2))

        chi2, p, dof, expected = chi2_contingency(cont_table)
        st.markdown("#### Test Chi-patrat:")
        st.write(f"Chi-patrat = {chi2:.4f}, p = {p:.4f}")
        if p < 0.05:
            st.success(
                f"Exista asociere semnificativa între {selected_cat} și diagnosticul (p < 0.05)."
            )
        else:
            st.info(
                f"Nu exista asociere semnificativa între {selected_cat} și diagnosticul (p ≥ 0.05)."
            )

        st.markdown("#### Interpretare:")
        if selected_cat == "Gender":
            st.markdown(
                "**Gender**: Difera riscul de diabet între sexe. Factor hormonal și comportamental."
            )
        elif selected_cat == "Ethnicity":
            st.markdown(
                "**Ethnicity**: Identifica grupuri etnice cu risc genetic mai mare."
            )
        elif selected_cat == "FamilyHistory":
            st.markdown("**FamilyHistory**: Confirma predispoziția genetica la diabet.")
        else:
            st.markdown(
                f"**{selected_cat}**: Reflecta comorbiditați (hipertensiune, dislipidemie) și terapiev."
            )


def generate_pairplot(df_clean: pd.DataFrame, coloane_numerice: list[str]):
    """
    Genereaza Pairplot pentru variabilele selectate (max 6) și coloreaza dupa 'Diagnosis'.
    """
    st.markdown(
        "<h2 style='text-align: center; color: blue;'>Grafic Pairplot</h2>",
        unsafe_allow_html=True,
    )
    st.warning(
        """
        Generarea unui pairplot complet poate fi costisitoare.
        Alegeți un subset de variabile (maxim 6).
        """
    )
    selected_vars = st.multiselect(
        "Selecteaza variabile pentru pairplot (max 6):",
        coloane_numerice,
        default=["Age", "BMI", "FastingBloodSugar", "HbA1c"],
    )
    if len(selected_vars) < 2:
        st.error("Selectați cel puțin 2 variabile.")
        return

    if len(selected_vars) > 6:
        st.warning("Poate dura mult timp pentru >6 variabile.")

    plot_data = df_clean[selected_vars + ["Diagnosis"]].copy()
    plot_data["Diagnostic"] = plot_data["Diagnosis"].map(
        {0: "Fara diabet", 1: "Cu diabet"}
    )

    fig = plt.figure(figsize=(12, 10))
    g = sns.pairplot(
        plot_data,
        hue="Diagnostic",
        palette={"Fara diabet": "green", "Cu diabet": "red"},
        diag_kind="kde",
    )
    g.fig.suptitle("Pairplot pentru variabilele selectate", y=1.02, fontsize=16)
    st.pyplot(g.fig)

    st.markdown(
        """
        ### Observații Pairplot:
        - **Diagona­la**: Distribuții univariate (KDE).
        - **Fiecare grafic**: Relații bivariate și posibile clustere între pacienți.
        - Culoarea arata diferența între grupurile cu și fara diabet.
        """
    )


def analyze_correlations(df_clean: pd.DataFrame, coloane_numerice: list[str]):
    """
    Afișeaza matricea de corelații (Pearson sau Spearman) și/sau doar corelațiile cu 'Diagnosis'.
    """
    st.markdown(
        "<h2 style='text-align: center; color: blue;'>Analiza corelațiilor</h2>",
        unsafe_allow_html=True,
    )

    corr_type = st.radio("Tip corelație:", ["Pearson", "Spearman"])
    if corr_type == "Pearson":
        corr_matrix = df_clean[coloane_numerice + ["Diagnosis"]].corr(method="pearson")
        method = "Pearson"
    else:
        corr_matrix = df_clean[coloane_numerice + ["Diagnosis"]].corr(method="spearman")
        method = "Spearman"

    view_option = st.radio("Vizualizare:", ["Toate corelațiile", "Corelații cu Diagnosis"])
    if view_option == "Corelații cu Diagnosis":
        diag_corr = corr_matrix["Diagnosis"].drop("Diagnosis").sort_values(ascending=False)
        diag_df = pd.DataFrame(
            {"Variabila": diag_corr.index, f"Corelație {method}": diag_corr.values}
        )
        st.markdown(f"#### Corelații {method} cu Diagnosis:")
        st.write(diag_df)

        fig, ax = plt.subplots(figsize=(12, 10))
        bars = ax.barh(diag_corr.index, diag_corr.values, color=plt.cm.RdBu_r(0.5 + diag_corr.values / 2))
        ax.set_xlabel(f"Corelație {method}")
        ax.set_title(f"Corelații {method} cu Diagnosis")
        ax.axvline(x=0, color="black", linestyle="--", alpha=0.3)
        for i, v in enumerate(diag_corr.values):
            ax.text(v + 0.01 if v >= 0 else v - 0.05, i, f"{v:.3f}", va="center")
        st.pyplot(fig)

    else:
        fig, ax = plt.subplots(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            mask=mask,
            vmin=-1,
            vmax=1,
            ax=ax,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(f"Matricea de corelație ({method})")
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown(
        """
        ### Observații:
        - **FastingBloodSugar & HbA1c**: Corelație foarte puternica (criteriu diagnostic).
        - **SystolicBP & DiastolicBP**: Corelație așteptata (tensiuni arteriale).
        - **CholesterolTotal & CholesterolLDL**: Corelație logica (LDL face parte din total).
        - Corelațiile ajuta la selectarea caracteristicilor pentru modele predictive.
        """
    )


def encode_categoricals(df_clean: pd.DataFrame):
    """
    Arata exemplu de One-Hot Encoding pe 'Ethnicity' și permite aplicarea pe întreg setul.
    """
    st.markdown(
        "<h2 style='text-align: center; color: blue;'>Encodarea variabilelor categorice</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        **Metode:**  
        1. LabelEncoding (pentru variabile binare: Gender, FamilyHistory)  
        2. One-Hot Encoding (pentru nominale: Ethnicity, Occupation)  
        """
    )

    st.markdown("#### Exemplu One-Hot Encoding (prima categorie: Ethnicity)")
    nominal_var = st.selectbox("Selecteaza variabila nominala:", ["Ethnicity"])
    dummies = pd.get_dummies(df_clean[nominal_var], prefix=nominal_var)
    example = pd.concat(
        [df_clean[nominal_var].head(10).reset_index(drop=True),
         dummies.head(10).reset_index(drop=True)],
        axis=1,
    )
    st.write(example)

    if st.button("Aplica One-Hot Encoding pe întreg setul"):
        df_encoded = df_clean.copy()
        df_encoded = pd.get_dummies(df_encoded, columns=[nominal_var], drop_first=True)
        st.success("One-Hot Encoding aplicat cu succes!")
        st.write(df_encoded.head())
        st.markdown(f"Numar coloane înainte: {df_clean.shape[1]}")
        st.markdown(f"Numar coloane dupa: {df_encoded.shape[1]}")
        new_cols = [c for c in df_encoded.columns if c not in df_clean.columns]
        st.markdown("#### Coloane noi adaugate:")
        st.write(new_cols)


def standardize_data(df_clean: pd.DataFrame, coloane_numerice: list[str]):
    """
    Standardizeaza toate coloanele numerice din df_clean (Z-score) și le stocheaza în df_std (global).
    Pentru comparație, utilizatorul alege doar câteva variabile care sa fie afișate înainte vs. dupa.
    """

    st.markdown(
        "<h2 style='text-align: center; color: blue;'>Standardizarea datelor numerice</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        Standardizare (Z-score):  
        $$z = \\frac{x - \\mu}{\\sigma}$$  
        - Fiecare coloana numerica va avea dupa transformare media = 0 și deviația standard = 1.  
        """
    )

    # Utilizatorul poate alege un subset din coloane pentru afișarea statisticilor
    vars_std = st.multiselect(
        "Selecteaza variabile pentru afișarea statisticilor înainte și dupa:",
        coloane_numerice,
        default=["Age", "BMI", "FastingBloodSugar", "HbA1c"],
    )
    if not vars_std:
        st.warning("Selecteaza cel puțin o variabila pentru afișare.")
        return

    # 1) Construim df_std pornind de la df_clean (un copy complet)
    df_std_loc = df_clean.copy()

    # 2) Aplicam StandardScaler pe toate coloanele numerice
    scaler = StandardScaler()
    df_std_loc[coloane_numerice] = scaler.fit_transform(df_clean[coloane_numerice])

    st.session_state.df_std = df_std_loc

    # 3) Afișam statistici doar pentru coloanele alese în vars_std
    st.markdown("#### Statistici înainte vs. dupa standardizare (pentru variabilele selectate):")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Înainte**")
        st.write(df_clean[vars_std].describe().loc[["mean", "std"]])
    with col2:
        st.markdown("**Dupa**")
        st.write(df_std_loc[vars_std].describe().loc[["mean", "std"]])

    # 4) Comparam distribuțiile pentru una dintre variabilele selectate
    st.markdown("#### Comparare distribuții:")
    std_var = st.selectbox("Alege variabila pentru grafic:", vars_std, key="std_viz")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df_clean[std_var], kde=True, ax=ax1, color="blue")
    ax1.set_title(f"Original: {std_var}")
    sns.histplot(df_std_loc[std_var], kde=True, ax=ax2, color="green")
    ax2.set_title(f"Standardizat: {std_var}")
    st.pyplot(fig)

    # 5) Exemplu numeric (primele 10 rânduri)
    st.markdown("#### Exemplu (primele 10 rânduri):")
    example_df = pd.DataFrame({
        f"{std_var} (original)": df_clean[std_var].head(10).values,
        f"{std_var} (standardizat)": df_std_loc[std_var].head(10).values,
    })
    st.dataframe(example_df)



def normalize_data(df_clean: pd.DataFrame, coloane_numerice: list[str]):
    """
    Permite normalizarea Min-Max a variabilelor selectate și afișeaza comparații.
    """
    st.markdown(
        "<h2 style='text-align: center; color: blue;'>Normalizarea datelor numerice</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        Normalizarea Min-Max:  
        $$x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$  
         - Pune valorile în [0, 1]  
         - Pastreaza forma distribuției  
        """
    )

    vars_norm = st.multiselect(
        "Selecteaza variabile pentru normalizare:",
        coloane_numerice,
        default=["Age", "BMI", "FastingBloodSugar", "HbA1c"],
        key="norm_vars",
    )
    if not vars_norm:
        st.warning("Selecteaza cel puțin o variabila.")
        return

    df_norm = df_clean.copy()
    minmax_scaler = MinMaxScaler()
    df_norm[vars_norm] = minmax_scaler.fit_transform(df_norm[vars_norm])

    st.markdown("### Statistici înainte vs. dupa:")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Înainte de normalizare:")
        st.write(df_clean[vars_norm].describe().loc[["min", "max"]])
    with col2:
        st.markdown("#### Dupa normalizare:")
        st.write(df_norm[vars_norm].describe().loc[["min", "max"]])

    st.markdown("### Comparare distribuții înainte vs. dupa:")
    norm_var = st.selectbox("Selecteaza variabila pentru vizualizare:", vars_norm, key="norm_viz")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    sns.histplot(df_clean[norm_var], kde=True, ax=ax1, color="blue")
    ax1.set_title(f"Original: {norm_var}")
    sns.histplot(df_norm[norm_var], kde=True, ax=ax2, color="orange")
    ax2.set_title(f"Normalizat: {norm_var}")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("#### Exemplu (primele 10 rânduri):")
    example_df = pd.DataFrame({
        f"{norm_var} (original)": df_clean[norm_var].head(10).values,
        f"{norm_var} (normalizat)": df_norm[norm_var].head(10).values,
    })
    st.write(example_df)

    st.markdown("### Comparare metode scalare:")
    if st.checkbox("Compara standardizare vs. normalizare"):
        var_cmp = st.selectbox("Variabila pentru comparație:", vars_norm, key="compare_scales")
        std_vals = StandardScaler().fit_transform(df_clean[[var_cmp]]).flatten()
        norm_vals = MinMaxScaler().fit_transform(df_clean[[var_cmp]]).flatten()
        robust_vals = RobustScaler().fit_transform(df_clean[[var_cmp]]).flatten()
        compare_df = pd.DataFrame({
            "Original": df_clean[var_cmp].values,
            "Standardizare (z-score)": std_vals,
            "Normalizare (min-max)": norm_vals,
            "Scalare robusta": robust_vals,
        })
        st.write(compare_df.head(10))
        fig, axs = plt.subplots(4, 1, figsize=(10, 16))
        sns.histplot(compare_df["Original"], kde=True, ax=axs[0], color="blue")
        axs[0].set_title(f"Original: {var_cmp}")
        sns.histplot(compare_df["Standardizare (z-score)"], kde=True, ax=axs[1], color="green")
        axs[1].set_title(f"Standardizare: {var_cmp}")
        sns.histplot(compare_df["Normalizare (min-max)"], kde=True, ax=axs[2], color="orange")
        axs[2].set_title(f"Normalizare: {var_cmp}")
        sns.histplot(compare_df["Scalare robusta"], kde=True, ax=axs[3], color="purple")
        axs[3].set_title(f"Robust Scale: {var_cmp}")
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown("#### Statistici comparative:")
        st.write(compare_df.describe())


def exploratory_analysis(df: pd.DataFrame):

    """
    Controleaza întreaga secțiune de Analiza Exploratorie împarțita pe tab-uri,
    folosind funcțiile modulare de mai sus.
    """
    df_clean = clean_initial_data(df)

    coloane_numerice = [
        "Age",
        "BMI",
        "AlcoholConsumption",
        "PhysicalActivity",
        "DietQuality",
        "SleepQuality",
        "SystolicBP",
        "DiastolicBP",
        "FastingBloodSugar",
        "HbA1c",
        "SerumCreatinine",
        "BUNLevels",
        "CholesterolTotal",
        "CholesterolLDL",
        "CholesterolHDL",
        "CholesterolTriglycerides",
        "FatigueLevels",
        "MedicalCheckupsFrequency",
        "MedicationAdherence",
        "HealthLiteracy",
    ]

    # Secțiuni în tab-uri:
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
        [
            "Analiza generala",
            "Analiza valori outlier",
            "Analiza distribuției",
            "Grafic pairplot",
            "Analiza corelațiilor",
            "Encodare categorice",
            "Standardizare",
            "Normalizare",
        ]
    )
    with tab1:
        descriptive_general(df_clean, coloane_numerice)

    with tab2:
        analyze_outliers(df_clean, coloane_numerice)

    with tab3:
        plot_numeric_distributions(df_clean, coloane_numerice)

    with tab4:
        generate_pairplot(df_clean, coloane_numerice)

    with tab5:
        analyze_correlations(df_clean, coloane_numerice)

    with tab6:
        encode_categoricals(df_clean)

    with tab7:

        standardize_data(df_clean, coloane_numerice)

    with tab8:
        normalize_data(df_clean, coloane_numerice)


# -------------------------------
# Funcția principala
# -------------------------------

def main():
    df_diabet = load_data()

    st.markdown(
        '<h1 style="font-size: 40px; text-align: center; color: #1E3A8A;">'
        "Analiza datelor despre diabet"
        "</h1>",
        unsafe_allow_html=True,
    )

    section = st.sidebar.radio(
        "Navigheaza la:",
        [
            "Detalii despre setul de date",
            "Analiza Exploratorie",
            "Analiza Clustering K-means",
            "Model de Regresie Liniara",
            "Model de Regresie (statsmodels)",
        ],
    )

    if section == "Detalii despre setul de date":
        show_dataset_details(df_diabet)

    elif section == "Analiza Exploratorie":
        exploratory_analysis(df_diabet)

    elif section == "Analiza Clustering K-means":
        page_analiza_clustering_diabet(df_diabet)

    elif section == "Model de Regresie Liniara":

        df_std_loc = st.session_state.df_std
        print(df_std_loc)
        page_model_regresie_diabet(df_std_loc)

    else:  # "Model de Regresie (statsmodels)"
        page_model_regresie_statsmodels(df_diabet)


if __name__ == "__main__":
    main()


# CSS suplimentar pentru personalizare stil
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
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
