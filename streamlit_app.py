import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io
from collections import Counter
import base64


# Set page configuration to include a title and icon (WCAG guideline 2.4.2)
st.set_page_config(
    page_title="Dashboard des sentiments associés aux tweets",
    page_icon=":bar_chart:",
    layout="wide"
)

# Custom CSS for high contrast and scalable text (WCAG guideline 1.4.3 & 1.4.4)
st.markdown(
    """
    <style>
    /* High contrast for text and background */
    body {
        color: #000000;
        background-color: #ffffff;
    }
    /* Use relative font sizes to allow text resizing */
    .stMarkdown, .stHeader, .stTitle {
        font-size: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

########################################################################################
# Load data: local file
########################################################################################
# Load data
# @st.cache_data
# def load_data(csv_path):
#     df = pd.read_csv(csv_path)
#     return df

# df = load_data('/Users/tgeof/Documents/Documents/B - Travaux Perso/1 - Scolarité/Ingenieur IA/OpenClassroom/_Projets/10. Développez une preuve de concept/2_project/data/processed/20250307-121318/test_df_with_preds.csv')

########################################################################################
# Load data: remote file
########################################################################################
@st.cache_data
def load_data():
    url = "https://github.com/Bitnair/oc_p10_poc/blob/main/test_df_with_preds.csv"
    df = pd.read_csv(url)
    return df


# Title & intro
st.title("Dashboard des sentiments associés aux tweets")
st.write(
    """
    Ce tableau de bord offre une vue interactive du jeu de données
    et affiche les sentiments prédits par 'bi-LSTM' et 'LLaMA 3 1B'.
    """
)

# Basic EDA
st.header("Analyse exploratoire des données (EDA)")

## Distribution of tweet lengths
tweet_lengths = df['tweet'].apply(lambda x: len(str(x).split()))
fig_len = px.histogram(
    x=tweet_lengths,
    nbins=30,
    labels={'x': 'Nombre de mots par tweet'},
    title='Distribution de la longueur des tweets'
)
fig_len.update_layout(
    xaxis_title="Nombre de mots",
    yaxis_title="Nombre",
)
st.plotly_chart(fig_len, use_container_width=True)
st.caption("Histogramme montrant la distribution de la longueur des tweets en nombre de mots.")

## Most frequent words
st.subheader("Top 20 des mots les plus fréquents")
all_words = []
for text in df['tweet']:
    all_words.extend(text.split())

counter = Counter(all_words)
top_n = 20
most_common = counter.most_common(top_n)
words, freqs = zip(*most_common)

fig_freq = px.bar(
    x=words,
    y=freqs,
    labels={'x': 'Mots', 'y': 'Fréquence'},
    title=f"Top {top_n} des mots les plus fréquents"
)
fig_freq.update_layout(
    xaxis_title="Mots",
    yaxis_title="Fréquence",
)
st.plotly_chart(fig_freq, use_container_width=True)
st.caption("Graphique en barres montrant les 20 mots les plus fréquents et leur fréquence dans le corpus.")

import base64

## WordCloud
st.subheader("Nuage de mots du jeu de données")
text_corpus = " ".join(str(txt) for txt in df['tweet'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_corpus)

# Create a matplotlib figure for the wordcloud
fig_wc, ax = plt.subplots(figsize=(8, 4))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
plt.tight_layout()

# Save the figure to a buffer
buf = io.BytesIO()
fig_wc.savefig(buf, format="png")
buf.seek(0)

# Convert buffer to base64 string
data = base64.b64encode(buf.read()).decode("utf-8")
html = f'''
<img src="data:image/png;base64,{data}" 
     alt="Nuage de mots montrant visuellement la fréquence des mots, où la taille de chaque mot reflète son occurrence." 
     style="width:100%; max-width:800px;">
'''
st.markdown(html, unsafe_allow_html=True)
st.caption("Nuage de mots de tous les tweets (visualisation de la fréquence du texte)")

# Prediction
st.header("Démo de prédiction")

# Option A: select from existing tweets
st.write("Sélectionnez un tweet existant dans le jeu de données pour voir les prédictions.")
tweet_index = st.selectbox(
    "Choisissez un tweet",
    options=range(len(df)),
    format_func=lambda x: f"Index {x} | {df['tweet'].iloc[x][:50]}..."
)

selected_tweet = df['tweet'].iloc[tweet_index]
st.markdown(f"**Tweet sélectionné :** {selected_tweet}")

# Show all predictions side by side
st.write("**Étiquette réelle :**", df['label'].iloc[tweet_index])
st.write("**Prédiction bi-LSTM :**", df['baseline_bilstm_pred'].iloc[tweet_index])
st.write("**Prédiction LLaMA Zero-Shot :**", df['llama_zeroshot_pred'].iloc[tweet_index])
st.write("**Prédiction LLaMA Fine-Tuned :**", df['llama_zeroshot_finetuned_pred'].iloc[tweet_index])

st.markdown("---")

# Accessibility notes (informational)
st.markdown("""
*Considérations d'accessibilité :*
- Toutes les images disposent de légendes ou de textes alternatifs pour les lecteurs d'écran.
- Le contenu non textuel, comme le nuage de mots, est accompagné d'un texte alternatif décrivant son contenu.
- La couleur n'est pas utilisée comme seul moyen de transmettre des informations, car des légendes et des descriptions textuelles sont fournies.
- Le contraste entre le texte et le fond respecte les recommandations (ratio d'au moins 4.5:1).
- Les textes peuvent être redimensionnés jusqu'à 200% sans perte de contenu ou de fonctionnalité.
- La page dispose d'un titre descriptif pour faciliter la navigation.
""")
