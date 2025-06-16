import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("resultados_entrenamiento_red_social_avanzada.csv")


fig, ax1 = plt.subplots(figsize=(10, 5))


ax1.set_xlabel('Época')
ax1.set_ylabel('Precisión', color='tab:blue')
ax1.plot(df['epoch'], df['accuracy'], color='tab:blue', label='Precisión')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_ylim(0, 1.05)


ax2 = ax1.twinx()
ax2.set_ylabel('Pérdida', color='tab:red')
ax2.plot(df['epoch'], df['loss'], color='tab:red', label='Pérdida')
ax2.tick_params(axis='y', labelcolor='tab:red')


plt.title("Comportamiento del modelo GCN - Red Social")
fig.tight_layout()
plt.savefig("entrenamiento_grafico_precision_perdida_avanzado.png")
plt.show()
