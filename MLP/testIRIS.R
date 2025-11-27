# ==================================================================
# ARQUIVO: testIRIS.R  
# DESCRIÇÃO: Testa o MLP com Iris completo [4, H, 3]
# ==================================================================

# pegar todos os pacotes instalados na máquina
df = data.frame(installed.packages())
pacotes = df$Package
if(!("ggplot2" %in% pacotes)) {
  install.packages("ggplot2")
} else {
  cat(" - ggplot2 já está instalado na máquina\n")
}

library(ggplot2)
source("mlp.R")

# Configuração
# seeds <- c(42, 123, 456)
seeds = 1:10
results = list()

cat("=== TESTE MLP COM IRIS (3 CLASSES) - VERSÃO CORRIGIDA ===\n")

# Preparar dados Iris
data(iris)
iris_data = iris
iris_data$class = as.numeric(factor(iris$Species)) - 1  # 0,1,2

# Normalizar features
normalize = function(x) { (x - min(x)) / (max(x) - min(x)) }
iris_data[, 1:4] = as.data.frame(lapply(iris_data[, 1:4], normalize))

# Codificar one-hot manualmente
iris_encoded = data.frame(
  Sepal.Length = iris_data$Sepal.Length,
  Sepal.Width = iris_data$Sepal.Width,
  Petal.Length = iris_data$Petal.Length,
  Petal.Width = iris_data$Petal.Width,
  class_0 = as.numeric(iris_data$class == 0),
  class_1 = as.numeric(iris_data$class == 1),
  class_2 = as.numeric(iris_data$class == 2)
)

cat("Dimensões do dataset:", dim(iris_encoded), "\n")
cat("Primeiras linhas:\n")
print(head(iris_encoded))

pdf("mlp_iris_comparison.pdf", width = 12, height = 10)

for(i in 1:length(seeds)) {
  cat("\n--- Execução", i, "com seed", seeds[i], "---\n")
  
  tryCatch({
    # MLP [4, 5, 3]
    model = mlp.create(input.length = 4, hidden.length = 5, output.length = 3)
    trained = mlp.train(model, iris_encoded, lrn.rate = 0.1, threshold = 0.05, n.iter = 1000)
    
    # Avaliar usando a nova função
    accuracy_result = mlp.accuracy(trained$model, iris_encoded[, 1:4], iris_data$class)
    
    results[[i]] = list(
      seed = seeds[i],
      accuracy = accuracy_result$accuracy,
      confusion_matrix = accuracy_result$confusion_matrix,
      trained = trained
    )
    
    cat("✅ Acurácia:", round(accuracy_result$accuracy, 4), "\n")
    
    # Gráfico (apenas se treinamento foi bem sucedido)
    if(length(trained$errorVec) > 0) {
      performance_data = data.frame(
        Epoca = 1:trained$epochs,
        Erro = trained$errorVec,
        Execucao = paste("Seed", seeds[i])
      )
      
      p = ggplot(performance_data, aes(x = Epoca, y = Erro, color = Execucao)) +
        geom_line() +
        labs(title = paste("MLP Iris - Seed", seeds[i]),
             subtitle = paste("Acurácia:", round(accuracy, 4)),
             x = "Época", y = "Erro Quadrático Médio") +
        theme_minimal()
      
      print(p)
    }
    
  }, error = function(e) {
    cat("❌ Erro na execução", i, ":", e$message, "\n")
  })
}

# Gráfico comparativo (apenas se temos resultados)
if(length(results) > 0) {
  comparison_data = data.frame()
  
  for(i in 1:length(results)) {
    if(!is.null(results[[i]])) {
      temp_data = data.frame(
        Epoca = 1:results[[i]]$trained$epochs,
        Erro = results[[i]]$trained$errorVec,
        Execucao = paste("Seed", seeds[i])
      )
      comparison_data = rbind(comparison_data, temp_data)
    }
  }
  
  if(nrow(comparison_data) > 0) {
    p_final = ggplot(comparison_data, aes(x = Epoca, y = Erro, color = Execucao)) +
      geom_line(alpha = 0.7) +
      labs(title = "Comparação MLP Iris - Múltiplas Seeds",
           x = "Época", y = "Erro Quadrático Médio") +
      theme_minimal()
    
    print(p_final)
  }
}

if(length(results) > 1) {
  # Encontrar execuções válidas
  valid_results = results[!sapply(results, is.null)]
  
  if(length(valid_results) > 1) {
    # Calcular número máximo de épocas
    max_epochs = max(sapply(valid_results, function(x) x$trained$epochs))
    
    # Criar matriz para armazenar todos os erros
    error_matrix = matrix(NA, nrow = max_epochs, ncol = length(valid_results))
    
    for(i in 1:length(valid_results)) {
      epochs = valid_results[[i]]$trained$epochs
      error_matrix[1:epochs, i] <- valid_results[[i]]$trained$errorVec
    }
    
    # Calcular média e desvio padrão
    mean_error = apply(error_matrix, 1, mean, na.rm = TRUE)
    sd_error = apply(error_matrix, 1, sd, na.rm = TRUE)
    
    # Criar dataframe para o gráfico
    ribbon_data = data.frame(
      Epoca = 1:max_epochs,
      Media = mean_error,
      SD = sd_error,
      Upper = mean_error + sd_error,
      Lower = pmax(mean_error - sd_error, 0)
    )
    
    # Gráfico com ribbon
    p_ribbon = ggplot(ribbon_data, aes(x = Epoca, y = Media)) +
      geom_ribbon(aes(ymin = Lower, ymax = Upper), 
                  fill = "lightgreen", alpha = 0.3) +
      geom_line(aes(color = "Média"), linewidth = 1.2) +
      geom_point(aes(color = "Média"), size = 0.8) +
      labs(title = "Curva de Aprendizado - Média e Desvio Padrão (Iris)",
           subtitle = paste("Média de", length(valid_results), "execuções com seeds diferentes"),
           x = "Época", 
           y = "Erro Quadrático Médio",
           color = "Legenda") +
      scale_color_manual(values = c("Média" = "darkgreen")) +
      theme_minimal() +
      theme(legend.position = "top")
    
    print(p_ribbon)
    
    cat("\n--- ESTATÍSTICAS DAS EXECUÇÕES IRIS ---\n")
    accuracies = sapply(valid_results, function(x) x$accuracy)
    epochs = sapply(valid_results, function(x) x$trained$epochs)
    
    cat("Número de execuções válidas:", length(valid_results), "\n")
    cat("Épocas médias:", round(mean(epochs), 1), "\n")
    cat("Acurácia média:", round(mean(accuracies), 4), "\n")
    cat("Desvio padrão da acurácia:", round(sd(accuracies), 4), "\n")
    cat("Acurácia mínima:", round(min(accuracies), 4), "\n")
    cat("Acurácia máxima:", round(max(accuracies), 4), "\n")
  }
}

if(length(results) > 1) {
  # Encontrar execuções válidas
  valid_results = results[!sapply(results, is.null)]
  
  if(length(valid_results) > 1) {
    # Combinar todos os dados em um dataframe
    all_data = data.frame()
    for(i in 1:length(valid_results)) {
      temp_data = data.frame(
        Epoca = 1:valid_results[[i]]$trained$epochs,
        Erro = valid_results[[i]]$trained$errorVec,
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
                  fill = "lightgreen", alpha = 0.3) +
      geom_line(color = "darkgreen", linewidth = 1.2) +
      geom_point(color = "darkgreen", size = 0.8, alpha = 0.6) +
      labs(title = "Curva de Aprendizado - Média e Desvio Padrão (Iris)",
           subtitle = paste("Média de", length(valid_results), "execuções com seeds diferentes"),
           x = "Época", 
           y = "Erro Quadrático Médio") +
      theme_minimal()
    
    print(p_ribbon)
    
    # Estatísticas descritivas
    cat("\n--- ESTATÍSTICAS DAS EXECUÇÕES IRIS ---\n")
    accuracies = sapply(valid_results, function(x) x$accuracy)
    epochs = sapply(valid_results, function(x) x$trained$epochs)
    
    cat("Número de execuções válidas:", length(valid_results), "\n")
    cat("Épocas médias:", round(mean(epochs), 1), "\n")
    cat("Acurácia média:", round(mean(accuracies), 4), "\n")
    cat("Desvio padrão da acurácia:", round(sd(accuracies), 4), "\n")
    cat("Acurácia mínima:", round(min(accuracies), 4), "\n")
    cat("Acurácia máxima:", round(max(accuracies), 4), "\n")
  }
}

dev.off()

# Resultados finais
cat("\n=== RESULTADOS FINAIS IRIS ===\n")
if(length(results) > 0) {
  for(i in 1:length(results)) {
    if(!is.null(results[[i]])) {
      cat("Seed", seeds[i], "- Acurácia:", round(results[[i]]$accuracy, 4), 
          "- Épocas:", results[[i]]$trained$epochs, "\n")
    }
  }
} else {
  cat("Nenhum resultado foi gerado devido a erros.\n")
}
