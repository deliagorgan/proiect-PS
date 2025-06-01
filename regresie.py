import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import streamlit as st


def page_model_regresie_diabet(df_diabet):
    st.title("Model de regresie liniară pentru predicția diabetului")

    st.markdown("""
    În această secțiune, vom construi un model de regresie liniară pentru a analiza factorii care influențează
    hemoglobina glicată (HbA1c) - un indicator esențial în diagnosticarea și monitorizarea diabetului.
    Înțelegerea factorilor care influențează nivelul HbA1c poate oferi informații valoroase pentru
    prevenția și managementul diabetului.
    """)
    print(df_diabet)


    if not isinstance(df_diabet, pd.DataFrame):
        st.error("Nu există date disponibile pentru analiză.")
        return df_diabet

    cols_to_drop = ['PatientID', 'DoctorInCharge'] if 'PatientID' in df_diabet.columns else []
    df_diabet_clean = df_diabet.drop(cols_to_drop, axis=1) if cols_to_drop else df_diabet.copy()

    target_variable = "HbA1c"

    if target_variable not in df_diabet_clean.columns:
        st.error(f"Variabila țintă '{target_variable}' nu există în setul de date.")
        return df_diabet

    st.subheader("1. Separarea datelor în caracteristici de intrare și țintă")

    st.markdown(f"""
    Pregătim datele pentru antrenarea modelului de regresie:
    - Variabila țintă: **{target_variable}** (Hemoglobina glicată - HbA1c)
    - Caracteristici de intrare: Vom selecta din setul de date variabilele relevante pentru model
    """)

    if len(df_diabet_clean) < 10:
        st.error("Nu există suficiente date pentru a crea un model de regresie.")
        return df_diabet

    y = df_diabet_clean[target_variable].values

    apply_log = False
    if y.max() > 1000 or y.var() > 10000:
        apply_log = st.checkbox("Aplică transformare logaritmică pentru variabila țintă", value=True)
        if apply_log:
            y = np.log(y + 1)  # Adăugăm 1 pentru a evita log(0)
            st.info("Am aplicat logaritmul natural pentru variabila țintă pentru a îmbunătăți performanța modelului.")

    numerical_columns = df_diabet_clean.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if 'Diagnosis' in numerical_columns:
        numerical_columns.remove('Diagnosis')
    if target_variable in numerical_columns:
        numerical_columns.remove(target_variable)

    predictors_to_avoid = ['AntidiabeticMedications', 'Diagnosis']
    for col in predictors_to_avoid:
        if col in numerical_columns:
            numerical_columns.remove(col)

    X_columns = st.multiselect(
        "Selectați caracteristicile pentru model:",
        numerical_columns,
        default=['BMI', 'Age', 'FastingBloodSugar', 'PhysicalActivity', 'DietQuality', 'SleepQuality']
    )

    if not X_columns:
        st.error("Trebuie să selectați cel puțin o caracteristică pentru model.")
        return df_diabet

    X = df_diabet_clean[X_columns]

    categorical_columns = X.select_dtypes(include=['object', 'category']).columns
    if not categorical_columns.empty:
        st.subheader("2. Codificarea variabilelor categorice")
        st.markdown("Următoarele variabile categorice vor fi codificate folosind one-hot encoding:")
        st.write(categorical_columns.tolist())

        X = pd.get_dummies(X)
        st.success(f"Codificare realizată cu succes. Noul set de date are {X.shape[1]} caracteristici.")

    st.subheader("2. Împărțirea datelor în seturi de antrenare și testare")

    st.markdown("""
    Urmează să împărțim setul de date în două subseturi: unul pentru antrenarea modelului și altul pentru testare. 
    Această separare este esențială în procesul de învățare automată, deoarece ne permite să evaluăm corect performanța modelului.

    Setul de antrenare permite modelului să învețe relațiile dintre variabile și să identifice tipare,
    în timp ce setul de testare ne ajută să verificăm cât de bine generalizează modelul pe date noi, nevăzute în timpul antrenării.
    Vom folosi 80% din date pentru antrenare și 20% pentru testare, un raport standard în domeniul învățării automate.
    """)

    test_size = 0.2
    random_state = 42

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    st.markdown(f"- Date pentru antrenare: {X_train.shape[0]} înregistrări ({(1 - test_size) * 100:.0f}%)")
    st.markdown(f"- Date pentru testare: {X_test.shape[0]} înregistrări ({test_size * 100:.0f}%)")

    st.subheader("3. Antrenarea modelului de regresie liniară")

    train_model = False
    if 'lr_model_diabet' not in st.session_state or st.button("Antrenează modelul"):
        train_model = True

    if train_model:
        with st.spinner("Antrenez modelul de regresie liniară..."):
            lr = linear_model.LinearRegression()
            model = lr.fit(X_train, y_train)



            st.session_state.lr_model_diabet = model
            st.session_state.X_train_diabet = X_train
            st.session_state.X_test_diabet = X_test
            st.session_state.y_train_diabet = y_train
            st.session_state.y_test_diabet = y_test
            st.session_state.target_variable_diabet = target_variable
            st.session_state.apply_log_diabet = apply_log

            y_train_pred = model.predict(X_train)



            st.write("X_train shape:", X_train.shape)
            st.write("X_test  shape:", X_test.shape)

            st.write("Coeficienți regresie liniară:")
            st.write(pd.DataFrame({
                'Caracteristică': X_train.columns,
                'Coeficient': model.coef_
            }).sort_values('Coeficient', ascending=False))

            st.write("Statistici HbA1c (train):", pd.Series(y_train).describe())
            st.write("Statistici HbA1c (test):", pd.Series(y_test).describe())

            st.write("Statistici HbA1c (train):", pd.Series(y_train).describe())
            st.write("Statistici HbA1c (test):", pd.Series(y_test).describe())

            st.subheader("4. Evaluarea modelului pe setul de antrenare")

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(y_train_pred, y_train, alpha=0.5)
            ax.set_xlabel('Valori prezise')
            ax.set_ylabel('Valori reale')
            ax.set_title('Comparație între valorile reale și cele prezise (set de antrenare)')

            min_val = min(y_train.min(), y_train_pred.min())
            max_val = max(y_train.max(), y_train_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r-')

            st.pyplot(fig)

            mse_train = mean_squared_error(y_train, y_train_pred)
            rmse_train = np.sqrt(mse_train)
            mae_train = mean_absolute_error(y_train, y_train_pred)
            r2_train = r2_score(y_train, y_train_pred)

            st.session_state.train_metrics_diabet = {
                'mse': mse_train,
                'rmse': rmse_train,
                'mae': mae_train,
                'r2': r2_train
            }

            st.markdown("**Metrici de performanță (set de antrenare):**")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MSE", f"{mse_train:.4f}")
            col2.metric("RMSE", f"{rmse_train:.4f}")
            col3.metric("MAE", f"{mae_train:.4f}")
            col4.metric("R²", f"{r2_train:.4f}")

            y_test_pred = model.predict(X_test)

            st.subheader("5. Evaluarea modelului pe setul de testare")

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(y_test_pred, y_test, alpha=0.5)
            ax.set_xlabel('Valori prezise')
            ax.set_ylabel('Valori reale')
            ax.set_title('Comparație între valorile reale și cele prezise (set de testare)')

            min_val = min(y_test.min(), y_test_pred.min())
            max_val = max(y_test.max(), y_test_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r-')

            st.pyplot(fig)

            mse_test = mean_squared_error(y_test, y_test_pred)
            rmse_test = np.sqrt(mse_test)
            mae_test = mean_absolute_error(y_test, y_test_pred)
            r2_test = r2_score(y_test, y_test_pred)

            st.session_state.test_metrics_diabet = {
                'mse': mse_test,
                'rmse': rmse_test,
                'mae': mae_test,
                'r2': r2_test
            }

            st.markdown("**Metrici de performanță (set de testare):**")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MSE", f"{mse_test:.4f}")
            col2.metric("RMSE", f"{rmse_test:.4f}")
            col3.metric("MAE", f"{mae_test:.4f}")
            col4.metric("R²", f"{r2_test:.4f}")

            with st.expander("Explicarea metricilor și interpretarea rezultatelor"):
                st.markdown("""
                ### Metrici de performanță:

                - **MSE (Mean Squared Error)**: Media pătratelor diferențelor dintre valorile reale și cele prezise.
                  - *Interpretare*: Valori mai mici indică o potrivire mai bună a modelului. MSE penalizează erorile mari mai sever decât cele mici.
                  - *Limită*: Nu are o scală fixă, depinde de datele analizate.

                - **RMSE (Root Mean Squared Error)**: Rădăcina pătrată a MSE. Este în aceleași unități ca variabila țintă.
                  - *Interpretare*: Reprezintă aproximativ "eroarea medie" în unitatea variabilei țintă. Un RMSE de 0.5 pentru HbA1c înseamnă că, în medie, predicțiile sunt cu ±0.5 puncte diferite de valorile reale.
                  - *Când este bun*: RMSE < 10% din intervalul valorilor variabilei țintă este considerat bun.

                - **MAE (Mean Absolute Error)**: Media valorilor absolute ale erorilor.
                  - *Interpretare*: Similar cu RMSE, dar tratează toate erorile în mod egal, indiferent de mărime.
                  - *Comparație cu RMSE*: Dacă MAE este semnificativ mai mic decât RMSE, înseamnă că există câteva erori mari ("outliers" în predicții).

                - **R² (Coeficient de determinare)**: Procentul din variația variabilei țintă explicat de model.
                  - *Interpretare*: 
                     - R² = 1.0: Modelul explică perfect variația (predicție perfectă)
                     - R² = 0.7: Modelul explică 70% din variația datelor
                     - R² = 0: Modelul nu este mai bun decât media simplă a datelor
                     - R² < 0: Modelul este mai rău decât media simplă (extrem de slab)
                  - *Evaluare calitativă*:
                     - R² > 0.9: Excelent
                     - R² între 0.7-0.9: Bun
                     - R² între 0.5-0.7: Moderat
                     - R² între 0.3-0.5: Slab
                     - R² < 0.3: Foarte slab

                ### Interpretarea diferenței între setul de antrenare și testare:

                - **Performanță similară**: Dacă metricile sunt apropiate pe ambele seturi, modelul generalizează bine.
                - **Performanță mult mai bună pe antrenare**: Indică supraadjustare (overfitting) - modelul "memorează" datele de antrenare dar nu generalizează.
                - **Performanță mult mai slabă pe testare**: Poate indica că setul de testare conține tipuri de date diferite față de cel de antrenare.

                ### Interpretarea graficelor:

                - **Puncte aproape de linia roșie**: Predicții bune
                - **Puncte răspândite aleatoriu**: Model slab
                - **Puncte formând un tipar (curba, grupuri)**: Relație nelineară care nu e capturată de model
                - **Puncte mai depărtate la capete**: Model care nu captează bine valorile extreme
                """)

            coefficients = pd.DataFrame({
                'Caracteristică': X.columns,
                'Coeficient': model.coef_
            }).sort_values(by='Coeficient', ascending=False)

            coefficients['Importanță (|Coeficient|)'] = np.abs(coefficients['Coeficient'])
            coefficients_abs = coefficients.sort_values(by='Importanță (|Coeficient|)', ascending=False)

            st.session_state.coefficients_diabet = coefficients
            st.session_state.coefficients_abs_diabet = coefficients_abs

            st.subheader("6. Importanța caracteristicilor")

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(x='Importanță (|Coeficient|)', y='Caracteristică', data=coefficients_abs.head(15), ax=ax)
            ax.set_title('Top caracteristici după importanță absolută')
            st.pyplot(fig)

            fig, ax = plt.subplots(figsize=(10, 8))
            top_pos_coeffs = coefficients.head(10)
            top_neg_coeffs = coefficients.tail(10)
            top_coeffs = pd.concat([top_pos_coeffs, top_neg_coeffs]).sort_values(by='Coeficient')
            sns.barplot(x='Coeficient', y='Caracteristică', data=top_coeffs, ax=ax)
            ax.set_title('Top 10 coeficienți pozitivi și negativi')
            ax.axvline(x=0, color='gray', linestyle='--')
            st.pyplot(fig)

            st.markdown("**Coeficienții modelului:**")
            st.dataframe(coefficients_abs)

            with st.expander("Interpretarea coeficienților"):
                st.markdown("""
                ### Cum să interpretezi coeficienții modelului:

                #### Magnitudinea coeficienților (importanța)
                - **Coeficienți cu valoare absolută mare**: Caracteristicile cu valorile cele mai mari (pozitive sau negative) au cel mai mare impact asupra predicției HbA1c.
                - **Coeficienți aproape de zero**: Aceste caracteristici au o influență minimă în model.

                #### Direcția coeficienților (semnul)
                - **Coeficienți pozitivi**: O creștere a acestei caracteristici conduce la o creștere a HbA1c. De exemplu, dacă "BMI" are un coeficient pozitiv, atunci cu cât BMI-ul este mai mare, cu atât HbA1c tinde să fie mai ridicat (conform modelului).
                - **Coeficienți negativi**: O creștere a acestei caracteristici conduce la o scădere a HbA1c. De exemplu, dacă "PhysicalActivity" are un coeficient negativ, atunci cu cât activitatea fizică este mai intensă, cu atât HbA1c tinde să fie mai scăzut.

                #### Atenție la interpretare:
                1. **Unități diferite**: Coeficienții reflectă și unitățile de măsură ale caracteristicilor. O caracteristică măsurată în mii va avea un coeficient mai mic decât una măsurată în unități, chiar dacă importanța reală este similară.
                2. **Colinearitate**: Dacă există caracteristici puternic corelate, interpretarea individuală a coeficienților poate fi înșelătoare.
                3. **Variabile dummy**: Pentru variabilele categorice codificate (one-hot encoding), coeficienții arată diferența față de categoria de referință.

                #### Exemplu de interpretare:
                Dacă "FastingBloodSugar" are un coeficient de 0.5, înseamnă că o creștere de 1 unitate (de exemplu, 1 mg/dL) în glicemia à jeun este asociată cu o creștere de 0.5 unități în HbA1c (presupunând că toate celelalte variabile rămân constante).
                """)

                st.markdown("### Interpretarea specifică a celor mai influente caracteristici:")

                top_pos = coefficients.head(3)
                top_neg = coefficients.tail(3).iloc[::-1]

                st.markdown("#### Caracteristici cu influență pozitivă:")
                for i, row in top_pos.iterrows():
                    st.markdown(
                        f"- **{row['Caracteristică']}** (coeficient: {row['Coeficient']:.4f}): O creștere de 1 unitate în această caracteristică este asociată cu o creștere de {row['Coeficient']:.4f} unități în HbA1c, menținând toate celelalte variabile constante.")

                st.markdown("#### Caracteristici cu influență negativă:")
                for i, row in top_neg.iterrows():
                    st.markdown(
                        f"- **{row['Caracteristică']}** (coeficient: {row['Coeficient']:.4f}): O creștere de 1 unitate în această caracteristică este asociată cu o scădere de {abs(row['Coeficient']):.4f} unități în HbA1c, menținând toate celelalte variabile constante.")

    elif 'lr_model_diabet' in st.session_state:
        st.subheader("4. Evaluarea modelului pe setul de antrenare")

        st.markdown("**Metrici de performanță (set de antrenare):**")
        train_metrics = st.session_state.train_metrics_diabet
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MSE", f"{train_metrics['mse']:.4f}")
        col2.metric("RMSE", f"{train_metrics['rmse']:.4f}")
        col3.metric("MAE", f"{train_metrics['mae']:.4f}")
        col4.metric("R²", f"{train_metrics['r2']:.4f}")

        st.subheader("5. Evaluarea modelului pe setul de testare")

        st.markdown("**Metrici de performanță (set de testare):**")
        test_metrics = st.session_state.test_metrics_diabet
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MSE", f"{test_metrics['mse']:.4f}")
        col2.metric("RMSE", f"{test_metrics['rmse']:.4f}")
        col3.metric("MAE", f"{test_metrics['mae']:.4f}")
        col4.metric("R²", f"{test_metrics['r2']:.4f}")

        with st.expander("Explicarea metricilor și interpretarea rezultatelor"):
            st.markdown("""
            ### Metrici de performanță:

            - **MSE (Mean Squared Error)**: Media pătratelor diferențelor dintre valorile reale și cele prezise.
              - *Interpretare*: Valori mai mici indică o potrivire mai bună a modelului. MSE penalizează erorile mari mai sever decât cele mici.
              - *Limită*: Nu are o scală fixă, depinde de datele analizate.

            - **RMSE (Root Mean Squared Error)**: Rădăcina pătrată a MSE. Este în aceleași unități ca variabila țintă.
              - *Interpretare*: Reprezintă aproximativ "eroarea medie" în unitatea variabilei țintă. Un RMSE de 0.5 pentru HbA1c înseamnă că, în medie, predicțiile sunt cu ±0.5 puncte diferite de valorile reale.
              - *Când este bun*: RMSE < 10% din intervalul valorilor variabilei țintă este considerat bun.

            - **MAE (Mean Absolute Error)**: Media valorilor absolute ale erorilor.
              - *Interpretare*: Similar cu RMSE, dar tratează toate erorile în mod egal, indiferent de mărime.
              - *Comparație cu RMSE*: Dacă MAE este semnificativ mai mic decât RMSE, înseamnă că există câteva erori mari ("outliers" în predicții).

            - **R² (Coeficient de determinare)**: Procentul din variația variabilei țintă explicat de model.
              - *Interpretare*: 
                 - R² = 1.0: Modelul explică perfect variația (predicție perfectă)
                 - R² = 0.7: Modelul explică 70% din variația datelor
                 - R² = 0: Modelul nu este mai bun decât media simplă a datelor
                 - R² < 0: Modelul este mai rău decât media simplă (extrem de slab)
              - *Evaluare calitativă*:
                 - R² > 0.9: Excelent
                 - R² între 0.7-0.9: Bun
                 - R² între 0.5-0.7: Moderat
                 - R² între 0.3-0.5: Slab
                 - R² < 0.3: Foarte slab

            ### Interpretarea diferenței între setul de antrenare și testare:

            - **Performanță similară**: Dacă metricile sunt apropiate pe ambele seturi, modelul generalizează bine.
            - **Performanță mult mai bună pe antrenare**: Indică supraadjustare (overfitting) - modelul "memorează" datele de antrenare dar nu generalizează.
            - **Performanță mult mai slabă pe testare**: Poate indica că setul de testare conține tipuri de date diferite față de cel de antrenare.

            ### Interpretarea graficelor:

            - **Puncte aproape de linia roșie**: Predicții bune
            - **Puncte răspândite aleatoriu**: Model slab
            - **Puncte formând un tipar (curba, grupuri)**: Relație nelineară care nu e capturată de model
            - **Puncte mai depărtate la capete**: Model care nu captează bine valorile extreme
            """)

        st.subheader("6. Importanța caracteristicilor")

        st.markdown("**Coeficienții modelului:**")
        st.dataframe(st.session_state.coefficients_abs_diabet)

        fig, ax = plt.subplots(figsize=(10, 8))
        coefficients = st.session_state.coefficients_diabet
        top_pos_coeffs = coefficients.head(10)
        top_neg_coeffs = coefficients.tail(10)
        top_coeffs = pd.concat([top_pos_coeffs, top_neg_coeffs]).sort_values(by='Coeficient')
        sns.barplot(x='Coeficient', y='Caracteristică', data=top_coeffs, ax=ax)
        ax.set_title('Top coeficienți pozitivi și negativi pentru predicția HbA1c')
        ax.axvline(x=0, color='gray', linestyle='--')
        st.pyplot(fig)

    if 'lr_model_diabet' in st.session_state:
        st.subheader("7. Predicție pentru un singur caz")

        st.markdown("""
        Acum că modelul nostru este antrenat și evaluat, putem folosi acest model pentru a face predicții
        individuale pentru nivelul HbA1c. Selectați un index din setul de testare pentru a vedea
        cum modelul prezice nivelul HbA1c pentru acel caz specific și cât de aproape este față de valoarea reală.
        """)

        selected_index = st.selectbox("Selectați un index:", st.session_state.X_test_diabet.index)

        if selected_index is not None:
            model = st.session_state.lr_model_diabet
            X_test = st.session_state.X_test_diabet
            y_test = st.session_state.y_test_diabet
            apply_log = st.session_state.apply_log_diabet

            single_prediction = model.predict(X_test.loc[selected_index].values.reshape(1, -1))[0]
            actual_value = y_test[X_test.index.get_loc(selected_index)]

            if apply_log:
                single_prediction_original = np.exp(single_prediction) - 1
                actual_value_original = np.exp(actual_value) - 1
                st.markdown(f"**Valoare prezisă (original):** {single_prediction_original:.4f}")
                st.markdown(f"**Valoare reală (original):** {actual_value_original:.4f}")
            else:
                st.markdown(f"**Valoare prezisă:** {single_prediction:.4f}")
                st.markdown(f"**Valoare reală:** {actual_value:.4f}")

            st.markdown("**Detaliile cazului:**")
            st.dataframe(X_test.loc[[selected_index]])

            st.markdown("#### Analiză detaliată a predicției:")
            error = abs(single_prediction - actual_value)
            error_percent = (error / actual_value) * 100 if actual_value != 0 else float('inf')

            if apply_log:
                error_original = abs(single_prediction_original - actual_value_original)
                error_percent_original = (
                                                     error_original / actual_value_original) * 100 if actual_value_original != 0 else float(
                    'inf')
                col1, col2 = st.columns(2)
                col1.metric("Eroare absolută", f"{error_original:.4f}")
                col2.metric("Eroare procentuală", f"{error_percent_original:.2f}%")
            else:
                col1, col2 = st.columns(2)
                col1.metric("Eroare absolută", f"{error:.4f}")
                col2.metric("Eroare procentuală", f"{error_percent:.2f}%")

            if error_percent < 5:
                st.success("🎯 Predicție excelentă! Eroarea este sub 5% din valoarea reală.")
            elif error_percent < 10:
                st.success("✅ Predicție bună. Eroarea este între 5-10% din valoarea reală.")
            elif error_percent < 20:
                st.warning("⚠️ Predicție acceptabilă. Eroarea este între 10-20% din valoarea reală.")
            else:
                st.error("❌ Predicție slabă. Eroarea depășește 20% din valoarea reală.")

    st.subheader("8. Interpretarea generală a modelului și implicații medicale")
    with st.expander("Interpretarea generală a rezultatelor"):
        st.markdown("""
        ### Cum să interpretezi rezultatele modelului de regresie pentru HbA1c:

        #### 1. Calitatea generală a modelului
        - **R² între 0.7-1.0**: Modelul este bun sau excelent și explică mare parte din variația datelor, ceea ce înseamnă că factorii selectați sunt puternic determinanți pentru nivelul HbA1c.
        - **R² între 0.5-0.7**: Modelul este acceptabil, dar sugerează că există și alți factori importanți care influențează HbA1c și nu sunt incluși în model.
        - **R² sub 0.5**: Modelul este slab, indicând că variabilele selectate explică doar parțial variația HbA1c.

        #### 2. Importanța factorilor pentru managementul diabetului

        **Factori cu coeficienți pozitivi semnificativi:**
        - Acești factori sunt asociați cu creșterea HbA1c și pot reprezenta factori de risc pentru control glicemic deficitar
        - Pacienții ar trebui sfătuiți să fie atenți la acești factori și să-i gestioneze corespunzător

        **Factori cu coeficienți negativi semnificativi:**
        - Acești factori sunt asociați cu scăderea HbA1c și pot reprezenta factori protectori pentru un control glicemic bun
        - Sunt potențiale ținte pentru intervenții terapeutice și schimbări în stilul de viață

        #### 3. Aplicații clinice:

        - **Screening și prevenție**: Identificarea persoanelor cu risc ridicat pe baza profilului factorilor
        - **Personalizarea intervențiilor**: Adaptarea recomandărilor și tratamentelor în funcție de factorii cu cel mai mare impact
        - **Monitorizare**: Urmărirea factorilor cheie pentru a anticipa modificări ale HbA1c
        - **Educație**: Informarea pacienților despre factorii care le influențează nivelul HbA1c

        #### 4. Limitări ale modelului:

        - **Relații de cauzalitate**: Asocierea statistică nu implică neapărat cauzalitate
        - **Variabile neobservate**: Pot exista factori importanți care nu au fost incluși în model
        - **Interacțiuni complexe**: Modelul liniar simplu nu captează interacțiunile complexe dintre factori
        - **Variabilitate individuală**: Răspunsul individual poate varia față de tendințele generale identificate de model
        """)

    if 'Diagnosis' in df_diabet_clean.columns:
        st.subheader("9. Compararea factorilor între pacienți cu și fără diabet")

        diagnosis_col = 'Diagnosis'
        df_diabet_pozitiv = df_diabet_clean[df_diabet_clean[diagnosis_col] == 1]
        df_diabet_negativ = df_diabet_clean[df_diabet_clean[diagnosis_col] == 0]

        st.markdown("### Comparație statistică pentru HbA1c:")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Pacienți fără diabet:")
            st.write(df_diabet_negativ[target_variable].describe())

        with col2:
            st.markdown("#### Pacienți cu diabet:")
            st.write(df_diabet_pozitiv[target_variable].describe())

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=df_diabet_clean, x=target_variable, hue=diagnosis_col,
                     kde=True, palette=['green', 'red'], element='step')
        ax.set_title(f'Distribuția {target_variable} în funcție de diagnostic')
        ax.set_xlabel(target_variable)
        ax.set_ylabel('Frecvență')
        ax.legend(['Fără diabet', 'Cu diabet'])
        st.pyplot(fig)

        if len(X_columns) > 0:
            st.markdown("### Relația dintre factorii selectați și HbA1c în funcție de diagnostic:")

            selected_factor = st.selectbox(
                "Selectați un factor pentru analiză:",
                X_columns
            )

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=df_diabet_clean, x=selected_factor, y=target_variable,
                            hue=diagnosis_col, palette=['green', 'red'], alpha=0.7)

            sns.regplot(data=df_diabet_negativ, x=selected_factor, y=target_variable,
                        scatter=False, ax=ax, color='green', line_kws={'linestyle': '--'})
            sns.regplot(data=df_diabet_pozitiv, x=selected_factor, y=target_variable,
                        scatter=False, ax=ax, color='red', line_kws={'linestyle': '--'})

            ax.set_title(f'Relația dintre {selected_factor} și {target_variable}')
            ax.legend(['Fără diabet', 'Cu diabet'])
            st.pyplot(fig)

            st.markdown("""
            **Interpretare:**

            Graficul de mai sus ilustrează relația dintre factorul selectat și HbA1c, diferențiată în funcție de diagnostic.
            - Punctele verzi și linia verde reprezintă pacienții fără diabet
            - Punctele roșii și linia roșie reprezintă pacienții cu diabet

            Observați:
            - Diferențele în nivelul general al HbA1c între cele două grupuri
            - Dacă pantele liniilor de regresie sunt diferite, acest lucru sugerează că factorul influențează diferit HbA1c în cele două grupuri
            - Împrăștierea punctelor în jurul liniei de regresie indică variabilitatea relației
            """)

    return df_diabet


def page_model_regresie_statsmodels(df_diabet):
    st.title("Model de regresie multiplă (OLS) cu statsmodels pentru HbA1c")

    st.markdown("""
    Vom folosi pachetul **statsmodels** pentru a construi un model de regresie liniară multiplă (Ordinary Least Squares) 
    care să explice variabila țintă **HbA1c** pe baza altor variabile din setul nostru de date despre diabet.
    """)

    if not isinstance(df_diabet, pd.DataFrame):
        st.error("Nu există date disponibile pentru analiză.")
        return

    # 1. Curățare inițială și definirea țintei
    cols_to_drop = ['PatientID', 'DoctorInCharge'] if 'PatientID' in df_diabet.columns else []
    df_clean = df_diabet.drop(cols_to_drop, axis=1) if cols_to_drop else df_diabet.copy()

    target_variable = "HbA1c"
    if target_variable not in df_clean.columns:
        st.error(f"Variabila țintă '{target_variable}' nu există în setul de date.")
        return

    if df_clean.shape[0] < 10:
        st.error("Nu există suficiente date pentru a construi modelul de regresie.")
        return

    st.subheader("1. Definirea setului X (predictori) și y (țintă)")

    y = df_clean[target_variable]
    numerical_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns.tolist()
    # Eliminăm din coloanele numerice pe cele care nu vrem să le folosim ca predictori:
    for col in ['Diagnosis', target_variable, 'AntidiabeticMedications']:
        if col in numerical_cols:
            numerical_cols.remove(col)

    if not numerical_cols:
        st.error("Nu există coloane numerice disponibile pentru a fi folosite ca predictori.")
        return

    # 2. Selectarea predictorilor de către utilizator
    X_columns = st.multiselect(
        "Selectați variabile numerice pentru regresie multiplă:",
        numerical_cols,
        default=numerical_cols[:5]  # primele 5 ca sugestie
    )

    if not X_columns:
        st.error("Trebuie să selectați cel puțin o variabilă predictor.")
        return

    X = df_clean[X_columns]

    # 3. Dacă există categorice în X_columns (nu ar trebui, pentru că am filtrat numeric), le putem codifica:
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        st.subheader("Codificarea variabilelor categorice")
        st.markdown("Vom aplica one-hot encoding pentru coloanele categorice selectate:")
        st.write(cat_cols)
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        st.success(f"Au fost create {X.shape[1]} caracteristici în urma one-hot encoding.")

    # 4. Împărțirea în seturi de antrenare și testare
    st.subheader("2. Împărțirea datelor în antrenare și testare")
    test_size = st.slider("Procent test (% din total)", min_value=10, max_value=50, value=20, step=5)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size / 100, random_state=42
    )
    st.markdown(f"- {X_train.shape[0]} rânduri pentru antrenare")
    st.markdown(f"- {X_test.shape[0]} rânduri pentru testare")

    # 5. Construirea modelului OLS cu statsmodels
    st.subheader("3. Antrenarea modelului OLS (statsmodels)")
    if st.button("Antrenează modelul OLS"):
        # Adăugăm constanta (intercept) manual
        X_train_const = sm.add_constant(X_train)
        X_test_const = sm.add_constant(X_test)

        model_ols = sm.OLS(y_train, X_train_const).fit()
        st.session_state.model_ols = model_ols
        st.session_state.X_test_const = X_test_const
        st.session_state.y_test = y_test

        # 5.1. Afișăm sumarul complet
        st.text(model_ols.summary())

        # 5.2. Extragem și afișăm coeficienții într-un DataFrame
        coef_df = pd.DataFrame({
            'Coeficient': model_ols.params,
            'StdErr': model_ols.bse,
            't-val': model_ols.tvalues,
            'P>|t|': model_ols.pvalues,
            'IC 2.5%': model_ols.conf_int().iloc[:, 0],
            'IC 97.5%': model_ols.conf_int().iloc[:, 1]
        })
        coef_df.index.name = 'Caracteristică'
        st.subheader("Coeficienți și statistici inferențiale")
        st.dataframe(coef_df)

        # 5.3. Predicții și metrici pe setul de test
        y_pred_test = model_ols.predict(X_test_const)
        mse_test = mean_squared_error(y_test, y_pred_test)
        r2_test = r2_score(y_test, y_pred_test)

        st.subheader("4. Performanța modelului pe setul de testare")
        st.write(f"- **MSE (test):** {mse_test:.4f}")
        st.write(f"- **R² (test):** {r2_test:.4f}")

        # 5.4. Grafic real vs prezis (test)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_pred_test, y_test, alpha=0.5, color='orange')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("Valori prezise")
        ax.set_ylabel("Valori reale")
        ax.set_title("Real vs. Prezis (set test)")
        st.pyplot(fig)

        st.success("Model OLS antrenat și evaluat cu succes!")

    elif 'model_ols' in st.session_state:
        # Dacă modelul a fost deja antrenat anterior, afișăm din nou rezultatele
        model_ols = st.session_state.model_ols
        X_test_const = st.session_state.X_test_const
        y_test = st.session_state.y_test

        st.text(model_ols.summary())

        coef_df = pd.DataFrame({
            'Coeficient': model_ols.params,
            'StdErr': model_ols.bse,
            't-val': model_ols.tvalues,
            'P>|t|': model_ols.pvalues,
            'IC 2.5%': model_ols.conf_int().iloc[:, 0],
            'IC 97.5%': model_ols.conf_int().iloc[:, 1]
        })
        coef_df.index.name = 'Caracteristică'
        st.dataframe(coef_df)

        y_pred_test = model_ols.predict(X_test_const)
        mse_test = mean_squared_error(y_test, y_pred_test)
        r2_test = r2_score(y_test, y_pred_test)

        st.write(f"- **MSE (test):** {mse_test:.4f}")
        st.write(f"- **R² (test):** {r2_test:.4f}")

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_pred_test, y_test, alpha=0.5, color='orange')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("Valori prezise")
        ax.set_ylabel("Valori reale")
        ax.set_title("Real vs. Prezis (set test)")
        st.pyplot(fig)
