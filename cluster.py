import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import streamlit as st


def page_analiza_clustering_diabet(df_diabet):
    st.title("Analiza de clustering K-means pentru datele despre diabet")

    st.markdown("""
    În această secțiune, vom aplica algoritmul K-means pentru a identifica grupuri (clustere) 
    în setul de date privind diabetul. Clusteringul poate ajuta la identificarea 
    tipologiilor de pacienți cu caracteristici similare și poate oferi perspective asupra
    factorilor care contribuie la apariția diabetului.
    """)

    if not isinstance(df_diabet, pd.DataFrame):
        st.error("Nu există date disponibile pentru analiză.")
        return df_diabet

    if len(df_diabet) < 10:
        st.error("Nu există suficiente date pentru analiza de clustering.")
        return df_diabet

    cols_to_drop = ['PatientID', 'DoctorInCharge'] if 'PatientID' in df_diabet.columns else []
    df_diabet_clean = df_diabet.drop(cols_to_drop, axis=1) if cols_to_drop else df_diabet.copy()

    st.subheader("1. Selectarea variabilelor pentru clustering")

    coloane_numerice = df_diabet_clean.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if 'Diagnosis' in coloane_numerice:
        coloane_numerice.remove('Diagnosis')

    st.markdown("""
    Pentru clustering, vom folosi doar variabile numerice. Selectați două variabile principale pentru 
    vizualizarea clusterelor în 2D. Ulterior, puteți selecta variabile 
    suplimentare pentru un clustering mai complex.
    """)

    col1, col2 = st.columns(2)
    with col1:
        var1 = st.selectbox(
            "Selectați prima variabilă:",
            coloane_numerice,
            index=coloane_numerice.index('BMI') if 'BMI' in coloane_numerice else 0
        )

    with col2:
        remaining_columns = [col for col in coloane_numerice if col != var1]
        var2 = st.selectbox(
            "Selectați a doua variabilă:",
            remaining_columns,
            index=remaining_columns.index('HbA1c') if 'HbA1c' in remaining_columns else 0
        )

    additional_vars = st.multiselect(
        "Selectați variabile suplimentare pentru clustering (opțional):",
        [col for col in coloane_numerice if col not in [var1, var2]],
        default=['Age',
                 'FastingBloodSugar'] if 'Age' in coloane_numerice and 'FastingBloodSugar' in coloane_numerice else []
    )

    selected_vars = [var1, var2] + additional_vars

    st.subheader("2. Pregătirea datelor pentru clustering")

    st.markdown("""
    Înainte de a aplica algoritmul K-means, datele vor fi scalate pentru a asigura 
    că toate variabilele contribuie în mod egal la analiză, indiferent de unitățile lor de măsură.
    """)

    df_cluster = df_diabet_clean[selected_vars].dropna()
    X_raw = df_cluster.values

    if len(X_raw) < 10:
        st.error("Nu există suficiente date fără valori lipsă pentru analiza de clustering.")
        return df_diabet

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    st.success(f"Datele au fost scalate cu succes. Forma datelor: {X.shape}")

    st.subheader("3. Determinarea numărului optim de clustere")

    st.markdown("""
    Pentru a determina numărul optim de clustere, vom folosi două metode:

    1. **Metoda Elbow (Cotului)**: Plotăm WCSS (Within-Cluster Sum of Squares) pentru 
       diferite valori ale k și căutăm "cotul" în grafic.
    2. **Scorul Silhouette**: Măsoară cât de similare sunt obiectele în propriul cluster 
       comparativ cu alte clustere.
    """)

    wcss = []
    silhouette_scores = []
    k_range = range(2, min(11, len(X) - 1))  # Limitează k la n-1

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, k in enumerate(k_range):
        status_text.text(f"Calculez pentru k = {k}...")
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

        labels = kmeans.labels_

        progress_bar.progress((i + 1) / len(k_range))

    status_text.text("Calcule finalizate!")

    fig_elbow, ax_elbow = plt.subplots(figsize=(10, 6))
    ax_elbow.plot(list(k_range), wcss, marker='o', linestyle='-', color='red')
    ax_elbow.set_xlabel('Număr de clustere')
    ax_elbow.set_ylabel('WCSS')
    ax_elbow.set_title('Metoda Elbow pentru determinarea numărului optim de clustere')
    ax_elbow.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig_elbow)

    st.markdown("""
    **Interpretarea metodei Elbow**: 
    Căutăm "cotul" în grafic, punctul unde adăugarea de clustere suplimentare nu reduce 
    semnificativ WCSS. Acesta este considerat numărul optim de clustere.
    """)
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(df_cluster)  # data e DataFrame-ul sau array-ul tău
        score = silhouette_score(df_cluster, labels)
        silhouette_scores.append(score)

    fig_silhouette, ax_silhouette = plt.subplots(figsize=(10, 6))
    ax_silhouette.plot(list(k_range), silhouette_scores, marker='o', linestyle='-', color='blue')
    ax_silhouette.set_xlabel('Număr de clustere')
    ax_silhouette.set_ylabel('Scor Silhouette')
    ax_silhouette.set_title('Scorul Silhouette pentru diferite numere de clustere')
    ax_silhouette.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig_silhouette)

    st.markdown("""
    **Interpretarea scorului Silhouette**:
    - ~1: Punctele sunt bine atribuite clusterelor lor
    - ~0: Punctele sunt la granița între clustere
    - < 0: Punctele sunt probabil atribuite greșit

    Căutăm valoarea k care maximizează scorul Silhouette.
    """)

    results_df = pd.DataFrame({
        'Număr clustere': list(k_range),
        'WCSS': wcss,
        'Scor Silhouette': silhouette_scores
    })
    st.dataframe(results_df)

    optimal_k_silhouette = k_range[silhouette_scores.index(max(silhouette_scores))]

    st.info(f"""
    **Recomandare**: Conform scorului Silhouette, numărul optim de clustere este: **{optimal_k_silhouette}**

    Notă: Metoda Elbow necesită o interpretare vizuală, căutați "cotul" în graficul WCSS.
    """)

    st.subheader("4. Aplicarea algoritmului K-means")

    n_clusters = st.slider(
        "Selectați numărul de clustere:",
        min_value=2,
        max_value=min(10, len(X) - 1),
        value=optimal_k_silhouette
    )

    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    df_cluster['Cluster'] = cluster_labels

    if 'Diagnosis' in df_diabet_clean.columns:
        df_cluster['Diagnosis'] = df_diabet_clean.loc[df_cluster.index, 'Diagnosis']

    final_silhouette = silhouette_score(X, cluster_labels)

    st.success(f"""
    Clusteringul a fost realizat cu succes!

    - Număr de clustere: {n_clusters}
    - Scorul Silhouette final: {final_silhouette:.4f}
    """)

    st.subheader("5. Vizualizarea clusterelor")

    viz_df = pd.DataFrame({
        var1: df_cluster[var1],
        var2: df_cluster[var2],
        'Cluster': df_cluster['Cluster']
    })

    centroids_scaled = kmeans.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)

    centroids_df = pd.DataFrame({
        var1: centroids_original[:, 0],
        var2: centroids_original[:, 1],
    })

    fig_clusters, ax_clusters = plt.subplots(figsize=(12, 8))

    palette = sns.color_palette("hls", n_clusters)

    scatter = sns.scatterplot(
        data=viz_df,
        x=var1,
        y=var2,
        hue='Cluster',
        palette=palette,
        s=100,
        alpha=0.7,
        ax=ax_clusters
    )

    sns.scatterplot(
        data=centroids_df,
        x=var1,
        y=var2,
        s=200,
        color='red',
        marker='X',
        edgecolor='black',
        linewidth=1,
        label='Centroizi',
        ax=ax_clusters
    )

    ax_clusters.set_title(f'Clustere K-means (k={n_clusters})', fontsize=16)
    ax_clusters.set_xlabel(var1, fontsize=14)
    ax_clusters.set_ylabel(var2, fontsize=14)
    ax_clusters.grid(True, linestyle='--', alpha=0.3)

    ax_clusters.legend(title='Cluster', title_fontsize=12)

    st.pyplot(fig_clusters)

    if 'Diagnosis' in df_cluster.columns:
        st.subheader("Distribuția diagnosticului de diabet în fiecare cluster")

        cluster_diagnosis = pd.crosstab(
            df_cluster['Cluster'],
            df_cluster['Diagnosis'],
            normalize='index'
        ) * 100

        if 1 in cluster_diagnosis.columns:
            cluster_diagnosis.rename(columns={0: 'Fără diabet (%)', 1: 'Cu diabet (%)'}, inplace=True)
        else:
            cluster_diagnosis['Cu diabet (%)'] = 0
            cluster_diagnosis.rename(columns={0: 'Fără diabet (%)'}, inplace=True)

        st.dataframe(cluster_diagnosis.round(2))

        fig_diag, ax_diag = plt.subplots(figsize=(12, 6))
        cluster_diagnosis.plot(kind='bar', stacked=True, ax=ax_diag, colormap='RdYlGn_r')
        ax_diag.set_title('Distribuția diagnosticului de diabet în fiecare cluster')
        ax_diag.set_xlabel('Cluster')
        ax_diag.set_ylabel('Procentaj (%)')
        ax_diag.legend(title='Diagnostic')

        for container in ax_diag.containers:
            ax_diag.bar_label(container, fmt='%.1f%%')

        st.pyplot(fig_diag)

    st.subheader("6. Profilul clusterelor")

    st.markdown("""
    Să analizăm caracteristicile fiecărui cluster pentru a înțelege ce tipuri de grupuri au fost identificate.
    """)

    cluster_profiles = df_cluster.groupby('Cluster')[selected_vars].mean()

    st.write("**Media variabilelor pentru fiecare cluster:**")
    st.dataframe(cluster_profiles)

    fig_profiles = plt.figure(figsize=(14, 8))
    ax_profiles = sns.heatmap(
        cluster_profiles,
        annot=True,
        cmap="YlGnBu",
        fmt=".2f",
        linewidths=.5
    )
    plt.title('Profilul clusterelor (valori medii)', fontsize=16)
    st.pyplot(fig_profiles)

    st.subheader("7. Interpretarea clusterelor")

    st.markdown("""
    În funcție de valorile medii ale variabilelor în fiecare cluster, putem interpreta și caracteriza fiecare grup:
    """)

    for i in range(n_clusters):
        st.markdown(f"##### Cluster {i}:")

        cluster_profile = cluster_profiles.loc[i]
        global_means = df_cluster[selected_vars].mean()

        diff_pct = ((cluster_profile - global_means) / global_means * 100).round(1)

        sorted_features = diff_pct.abs().sort_values(ascending=False)

        top_features = min(5, len(sorted_features))
        distinctive_features = sorted_features.index[:top_features]

        for feature in distinctive_features:
            value = cluster_profile[feature]
            diff = diff_pct[feature]
            direction = "mai mare" if diff > 0 else "mai mic"

            st.markdown(f"- **{feature}**: {value:.2f} ({abs(diff):.1f}% {direction} decât media)")

        cluster_size = (df_cluster['Cluster'] == i).sum()
        cluster_pct = (cluster_size / len(df_cluster) * 100).round(1)

        st.markdown(f"- **Mărime**: {cluster_size} pacienți ({cluster_pct}% din total)")

        if 'Diagnosis' in df_cluster.columns:
            diag_in_cluster = df_cluster[df_cluster['Cluster'] == i]['Diagnosis'].mean() * 100
            st.markdown(f"- **Proporție pacienți cu diabet**: {diag_in_cluster:.1f}%")

        st.markdown("---")

    st.subheader("8. Export date cu clustere")

    if st.button("Pregătește datele cu etichete de cluster pentru export"):
        df_export = df_diabet.copy()

        cluster_mapping = dict(zip(df_cluster.index, df_cluster['Cluster']))
        df_export['Cluster'] = df_export.index.map(lambda x: cluster_mapping.get(x, np.nan))

        csv = df_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descarcă CSV cu clustere",
            data=csv,
            file_name="date_diabet_cu_clustere.csv",
            mime="text/csv"
        )

    st.subheader("9. Concluzii și aplicații medicale")

    st.markdown("""
    ### Utilitatea analizei de clustering pentru datele despre diabet:

    1. **Segmentarea pacienților**: Identificarea unor tipologii distincte de pacienți 
       în funcție de caracteristicile clinice și de stil de viață.

    2. **Intervenții personalizate**: Dezvoltarea unor strategii specifice de management 
       al diabetului pentru fiecare segment de pacienți.

    3. **Identificarea factorilor de risc**: Înțelegerea mai bună a combinațiilor de factori 
       care contribuie la apariția diabetului în diferite grupuri de pacienți.

    4. **Screening țintit**: Direcționarea eforturilor de screening către persoanele 
       care prezintă caracteristici similare cu clusterele cu risc ridicat.

    5. **Cercetare medicală**: Identificarea de subpopulații pentru studii clinice 
       mai specifice privind prevenția și tratamentul diabetului.
    """)

    if st.checkbox("Adaugă etichetele de cluster la setul de date pentru analize ulterioare"):
        df_result = df_diabet.copy()

        cluster_mapping = dict(zip(df_cluster.index, df_cluster['Cluster']))
        df_result['Cluster'] = df_result.index.map(lambda x: cluster_mapping.get(x, np.nan))

        return df_result
    else:
        return df_diabet