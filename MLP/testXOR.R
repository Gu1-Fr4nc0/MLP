# ==================================================================
# ARQUIVO: testXOR.R
# DESCRIÇÃO: Testa o MLP com dataset XOR [2, H, 1]
# ==================================================================

# pegar todos os pacotes instalados na máquina
df = data.frame(installed.packages())
pacotes = df$Package
if(!("ggplot2" %in% pacotes)) {
  install.packages("ggplot2")
} else {
  cat(" - ggplot2 já está instalado na máquina\n")
}

# Verificar e instalar dplyr se necessário
if(!("dplyr" %in% pacotes)) {
  install.packages("dplyr")
  cat(" - dplyr instalado\n")
} else {
  cat(" - dplyr já está instalado\n")
}

library(ggplot2)
library(dplyr)
source("mlp.R")

# Configuração de seeds
seeds = c(42, 123, 456, 789, 999)
results = list()

cat("=== TESTE MLP COM XOR ===\n")

# Dataset XOR
xor_data = data.frame(
  x1 = c(0, 0, 1, 1),
  x2 = c(0, 1, 0, 1),
  class = c(0, 1, 1, 0)
)

pdf("mlp_xor_comparison.pdf", width = 12, height = 10)

for(i in 1:length(seeds)) {
  cat("\n--- Execução", i, "com seed", seeds[i], "---\n")
  
  # Criar e treinar MLP [2, 4, 1]
  model = mlp.create(input.length = 2, hidden.length = 4, output.length = 1)
  trained = mlp.train(model, xor_data, lrn.rate = 0.3, threshold = 0.01, n.iter = 5000)
  
  # Fazer predições
  accuracy_result = mlp.accuracy(trained$model, xor_data, xor_data$class)
  
  results[[i]] = list(
    seed = seeds[i],
    model = trained$model,
    trained = trained,
    accuracy = accuracy_result$accuracy,
    confusion_matrix = accuracy_result$confusion_matrix,
    predictions = accuracy_result$predicted_classes
  )
  
  # Gráfico de performance
  performance_data = data.frame(
    Epoca = 1:trained$epochs,
    Erro = trained$errorVec,
    Execucao = paste("Seed", seeds[i])
  )
  
  p = ggplot(performance_data, aes(x = Epoca, y = Erro, color = Execucao)) +
    geom_line() +
    labs(title = paste("MLP XOR - Seed", seeds[i]),
         subtitle = paste("Acurácia:", round(accuracy, 4)),
         x = "Época", y = "Erro Quadrático Médio") +
    theme_minimal()
  
  print(p)
}

# Gráfico comparativo final
comparison_data = data.frame()
for(i in 1:length(results)) {
  temp_data = data.frame(
    Epoca = 1:results[[i]]$trained$epochs,
    Erro = results[[i]]$trained$errorVec,
    Execucao = paste("Seed", seeds[i])
  )
  comparison_data = rbind(comparison_data, temp_data)
}

p_final = ggplot(comparison_data, aes(x = Epoca, y = Erro, color = Execucao)) +
  geom_line(alpha = 0.7) +
  labs(title = "Comparação MLP XOR - Múltiplas Seeds",
       x = "Época", y = "Erro Quadrático Médio") +
  theme_minimal()

print(p_final)

# Resultados finais
cat("\n=== RESULTADOS FINAIS XOR ===\n")
for(i in 1:length(results)) {
  cat("Seed", seeds[i], "- Acurácia:", round(results[[i]]$accuracy, 4), 
      "- Épocas:", results[[i]]$trained$epochs, "\n")
}

if(length(results) > 1) {
  # Combinar todos os dados em um dataframe
  all_data = data.frame()
  for(i in 1:length(results)) {
    temp_data = data.frame(
      Epoca = 1:results[[i]]$trained$epochs,
      Erro = results[[i]]$trained$errorVec,
      Execucao = paste("Seed", seeds[i])
    )
    all_data = rbind(all_data, temp_data)
  }
  
  # Calcular média e desvio padrão por época
  summary_data = all_data %>%
    group_by(Epoca) %>%
    summarise(
      Media = mean(Erro),
      SD = sd(Erro),
      .groups = 'drop'
    ) %>%
    mutate(
      Upper = Media + SD,
      Lower = pmax(Media - SD, 0)
    )
  
  # Gráfico com ribbon
  p_ribbon = ggplot(summary_data, aes(x = Epoca, y = Media)) +
    geom_ribbon(aes(ymin = Lower, ymax = Upper), 
                fill = "lightblue", alpha = 0.3) +
    geom_line(color = "blue", linewidth = 1.2) +
    geom_point(color = "blue", size = 0.8, alpha = 0.6) +
    labs(title = "Curva de Aprendizado - Média e Desvio Padrão (XOR)",
         subtitle = paste("Média de", length(results), "execuções com seeds diferentes"),
         x = "Época", 
         y = "Erro Quadrático Médio") +
    theme_minimal()
  
  print(p_ribbon)
  
  # Estatísticas descritivas
  cat("\n--- ESTATÍSTICAS DAS EXECUÇÕES XOR ---\n")
  accuracies = sapply(results, function(x) x$accuracy)
  epochs = sapply(results, function(x) x$trained$epochs)
  
  cat("Número de execuções:", length(results), "\n")
  cat("Épocas médias:", round(mean(epochs), 1), "\n")
  cat("Acurácia média:", round(mean(accuracies), 4), "\n")
  cat("Desvio padrão da acurácia:", round(sd(accuracies), 4), "\n")
  cat("Acurácia mínima:", round(min(accuracies), 4), "\n")
  cat("Acurácia máxima:", round(max(accuracies), 4), "\n")
}

dev.off()
