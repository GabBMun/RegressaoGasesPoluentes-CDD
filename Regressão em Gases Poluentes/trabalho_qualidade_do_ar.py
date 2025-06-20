import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tkinter import Tk, ttk, Label
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Caminho relativo ao arquivo
caminho_arquivo = os.path.join(os.path.dirname(__file__), 'AirQualityUCI.xlsx')

# Carregando os dados
df = pd.read_excel(caminho_arquivo, sheet_name='AirQualityUCI')
df_numeric = df.select_dtypes(include='number').replace(-200, pd.NA)
df_clean = df_numeric.dropna()

# Correlação
correlation = df_clean.corr()
mean_corr = correlation.abs().mean().sort_values(ascending=False)

# Regressão linear
X = df_clean.drop(columns=['PT08.S2(NMHC)'])
y = df_clean['PT08.S2(NMHC)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

# Criar GUI com tkinter
janela = Tk()
janela.title("Análise de Qualidade do Ar")
janela.geometry("600x400")

# Label para R²
label_r2 = Label(janela, text=f"R² da regressão: {r2:.4f}", font=("Arial", 12, "bold"))
label_r2.pack(pady=10)

# Tabela com média da correlação
tabela = ttk.Treeview(janela, columns=("Variável", "Correlação Média"), show="headings")
tabela.heading("Variável", text="Variável")
tabela.heading("Correlação Média", text="Correlação Média")

for var, valor in mean_corr.items():
    tabela.insert("", "end", values=(var, f"{valor:.4f}"))

tabela.pack(expand=True, fill='both', padx=10, pady=10)

# Botão para exibir os gráficos após fechar a janela
def mostrar_graficos():
    janela.destroy()

    # Matriz de correlação
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Matriz de Correlação")

    # Regressão: valores reais vs previstos
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.xlabel("Valor real de PT08.S2(NMHC)")
    plt.ylabel("Valor previsto")
    plt.title("Valores reais vs previstos da regressão")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideal')
    plt.legend()
    plt.grid(True)

    # Histograma de resíduos
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, color="purple", bins=30)
    plt.title("Distribuição dos resíduos da regressão")
    plt.xlabel("Valor real - previsto")
    plt.ylabel("Frequência")
    plt.grid(True)

    # Histograma de benzeno
    plt.figure(figsize=(8, 6))
    sns.histplot(df_clean['C6H6(GT)'], bins=30, kde=True, color='teal')
    plt.title("Distribuição da concentração de benzeno (C6H6)")
    plt.xlabel("C6H6(GT) – µg/m³")
    plt.ylabel("Frequência")
    plt.grid(True)

    plt.show()

# Botão
ttk.Button(janela, text="Mostrar Gráficos", command=mostrar_graficos).pack(pady=10)

janela.mainloop()
