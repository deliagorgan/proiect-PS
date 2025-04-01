import pandas as pd
import seaborn as sns
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
df_diabet = pd.read_csv("./data/diabetes_data.csv")


st.markdown('<h1 style="color: inherit; font-size: 40px; text-align: center;">Analiza datelor </h1>', unsafe_allow_html=True)

section = st.sidebar.radio("Navigați la:",
                           ["Detalii despre setul de date", "Analiza Exploratorie"])

# ---------------------------
# Secțiunea: Detalii despre setul de date
# ---------------------------
if section == "Detalii despre setul de date":
    st.write(df_diabet)


    st.subheader("Variabilele setului de date:\n")
    with open("explicare_variabile.txt", "r") as file:
        continut = file.read()
    st.markdown(continut)



    numar_linii = df_diabet.shape[0]
    print(numar_linii)

    df_diabet = df_diabet.drop_duplicates()

    numar_linii = df_diabet.shape[0]
    print(numar_linii)

    st.markdown("Setul de date nu contine inregistrari duplicate!")


if section == "Analiza Exploratorie":
    st.markdown("Am eliminat coloanele care nu influenteaza analiza, adica PatientID si DoctorInCharge")
    cols_to_drop = ['PatientID', 'DoctorInCharge']
    df_diabet = df_diabet.drop(cols_to_drop, axis=1)
    st.write(df_diabet)

    coloane_numerice = ["Age", "BMI", "AlcoholConsumption", "PhysicalActivity", "DietQuality", "SleepQuality",
                        "SystolicBP", "DiastolicBP", "FastingBloodSugar", "HbA1c", "SerumCreatinine",
                        "BUNLevels", "CholesterolTotal", "CholesterolLDL", "CholesterolHDL", "CholesterolTriglycerides",
                        "FatigueLevels", "MedicalCheckupsFrequency", "MedicationAdherence", "HealthLiteracy"]


    coloane_categorice = df_diabet.columns.difference(coloane_numerice).tolist()
    df_poz_diabet = df_diabet[df_diabet['Diagnosis'] == 1]
    df_neg_diabet = df_diabet[df_diabet['Diagnosis'] == 0]

    tab_analiza_generala, tab_outlieri, tab_analiza_distributie, tab_grafic_pairplot, tab_analiza_corelatii,tab_encodare, tab_standardizare, tab_normalizare = st.tabs(["Analiza generala", "Analiza valori outlier", "Analiza distributiei datelor", "Grafic pairplot", "Analiza corelatiilor dintre variabile", "Encodarea variablelor discrete", "Standardizarea setului de date", "Normalizarea setului de date"])
    with tab_analiza_generala:
        st.markdown(
            "<h2 style='text-align: center; color: blue;'>Analiza generala a datelor numerice pentru pacientii</h2>",
            unsafe_allow_html=True
        )
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
            "<h2 style='text-align: center; color: red;'> au fost diagnosticati cu diabet <br> </h2>",
            unsafe_allow_html=True
            )
            st.write(df_poz_diabet[coloane_numerice].describe())

        with col2:
            st.markdown(
                "<h2 style='text-align: center; color: green;'>NU au fost diagnosticati cu diabet</h2>",
                unsafe_allow_html=True
            )

            st.write(df_neg_diabet[coloane_numerice].describe())




        nan_summary = df_diabet.isnull().sum()
        nan_percentage = (df_diabet.isnull().sum() / len(df_diabet)) * 100
        print(nan_percentage)
        st.markdown(
            "<h2 style='text-align: center; color: red;'>Interpretarea setului de date  </h2>",
            unsafe_allow_html=True
        )
        st.markdown(
            """ 

            Acest set de date conține **1879 de observații** și măsoară diverse variabile legate de sanatatea pacientilor care fie sufera, fie nu de diabet.

            ---
            """,
        unsafe_allow_html=True)
        col3, col4 = st.columns(2)
        with col3:
            with open("interpretari_pers_cu_diabet.txt", "r") as file:
                continut = file.read()
            st.markdown(continut,
                unsafe_allow_html=True
            )

        with col4:
            with open("interpretari_pers_fara_diabet.txt", "r") as file:
                continut = file.read()
            st.markdown(continut,
                unsafe_allow_html=True
            )

    with tab_analiza_distributie:
        st.markdown(
            "<h2 style='text-align: center; color: red;'>Analiza distributiei pentru persoanele care sufera de diabet</h2>",
            unsafe_allow_html=True
        )

        nr_grafice_rand = 2
        nr_randuri = int(np.ceil(len(coloane_numerice) / nr_grafice_rand))
        fig, axes = plt.subplots(nrows=nr_randuri, ncols=nr_grafice_rand, figsize=(6 * nr_grafice_rand, 4 * nr_randuri))

        axes = axes.flatten()  # Transformă subgraficile într-un vector 1D pentru indexare ușoară

        for i, col in enumerate(coloane_numerice):
            axes[i].hist(df_poz_diabet[col], edgecolor='black', color='skyblue', bins = 30)
            axes[i].set_title("Distribuție " + col)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Frecvență")
        plt.tight_layout()
        st.pyplot(fig)



        st.markdown(
            "<h2 style='text-align: center; color: red;'>Analiza distributiei pentru persoanele care NU sufera de diabet</h2>",
            unsafe_allow_html=True
        )


        nr_grafice_rand = 2
        nr_randuri = int(np.ceil(len(coloane_numerice) / nr_grafice_rand))
        fig, axes = plt.subplots(nrows=nr_randuri, ncols=nr_grafice_rand, figsize=(6 * nr_grafice_rand, 4 * nr_randuri))

        axes = axes.flatten()  # Transformă subgraficile într-un vector 1D pentru indexare ușoară

        for i, col in enumerate(coloane_numerice):
            axes[i].hist(df_neg_diabet[col], edgecolor='black', color='skyblue', bins = 30)
            axes[i].set_title("Distribuție " + col)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Frecvență")
        plt.tight_layout()
        st.pyplot(fig)

        plt.figure(figsize=(10, 6))
        sns.countplot(x='Ethnicity', data=df_diabet)
        plt.title('Ethnicity Distribution')
        plt.xlabel('Ethnicity')
        plt.ylabel('Count')
        plt.tight_layout()
        st.pyplot(plt)

        # Antihypertensive Medications
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 3, 1)
        sns.countplot(data=df_diabet, x='AntihypertensiveMedications', palette='Set1')
        plt.title('Use of Antihypertensive Medications')
        plt.xlabel('Antihypertensive Medications (0 = No, 1 = Yes)')
        plt.ylabel('Count')
        plt.tight_layout()
        # Statins
        plt.subplot(1, 3, 2)
        sns.countplot(data=df_diabet, x='Statins', palette='Set2')
        plt.title('Use of Statins')
        plt.xlabel('Statins (0 = No, 1 = Yes)')
        plt.ylabel('Count')
        plt.tight_layout()

        # Antidiabetic Medications
        plt.subplot(1, 3, 3)
        sns.countplot(data=df_diabet, x='AntidiabeticMedications', palette='Set3')
        plt.title('Use of Antidiabetic Medications')
        plt.xlabel('Antidiabetic Medications (0 = No, 1 = Yes)')
        plt.ylabel('Count')
        plt.tight_layout()
        st.pyplot(plt)

    with tab_outlieri:
        st.markdown(
            "<h2 style='text-align: center; color: red;'>Analiza valorilor extreme pentru toate valorile numerice</h2>",
            unsafe_allow_html=True
        )
        for column in coloane_numerice:
            st.markdown(f"#### *Analiza outlierilor pentru variabila {column}*")

            Q1 = df_diabet[column].quantile(0.25)
            Q3 = df_diabet[column].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = df_diabet[(df_diabet[column] < lower_bound) | (df_diabet[column] > upper_bound)]

            st.write(f"Limita inferioară: {lower_bound}")
            st.write(f"Limita superioară: {upper_bound}")
            st.write(f"Număr de outlieri: {len(outliers)}")

            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(y=df_diabet[column], ax=ax, color="skyblue")
            ax.set_title(f'Boxplot pentru {column}')
            st.pyplot(fig)

            if not outliers.empty:
                st.write("Valorile outlier:")
                st.write(outliers[[column, 'Gender', 'Age', 'Occupation', 'Sleep Disorder']])

    with tab_grafic_pairplot:
        st.markdown(
            "<h2 style='text-align: center; color: red;'>Analiza graficului pairplot</h2>",
            unsafe_allow_html=True
        )

        sns.pairplot(df_diabet[coloane_numerice], diag_kind="kde")
        plt.suptitle("Pairplot pentru variabilele numerice", y=1.02)
        st.pyplot(plt)

    with tab_analiza_corelatii:
        st.markdown(
            "<h2 style='text-align: center; color: red;'>Analiza corelatiilor dintre variabilele numerice</h2>",
            unsafe_allow_html=True
        )
        matrice_corelatie = df_diabet[coloane_numerice].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrice_corelatie, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Matricea de corelatie pentru valorile numerice")
        plt.tight_layout()
        st.pyplot(plt)

        st.markdown(
            "<h4 style='text-align: center; color: red;'>Interpretare matrice de coralatii</h4>",
            unsafe_allow_html=True
        )
        st.markdown(
            """
            Se remarca faptul ca valorile numerice nu sunt corelate intre ele. 
            """
        )

    with tab_encodare:
        st.markdown(
            """
            Singura variabila din setul nostru de date care este discreta si nu reprezinta o ordine naturala intre ele (low, medium, high)
            """
        )


    with tab_standardizare:
        scaler = StandardScaler()
        df_diabet[coloane_numerice] = scaler.fit_transform(df_diabet[coloane_numerice])

        st.write((df_diabet[coloane_numerice].describe()).loc[["mean", "std"]])
        st.markdown (
            """
            Dupa scalarea datelor media este : 
            """
        )

    with tab_normalizare:


        minmaxscaler = MinMaxScaler()
        df_diabet[coloane_numerice] = minmaxscaler.fit_transform(df_diabet[coloane_numerice])

        st.write((df_diabet[coloane_numerice].describe()).loc[["min", "max"]])
