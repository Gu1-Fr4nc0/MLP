# 🧠 MLP - Implementação Proprietária de Rede Neural Multicamadas

## 📋 Descrição do Projeto
Este projeto contém uma implementação completa de um **Perceptron Multicamadas (MLP)** desenvolvida do zero em **R**, como parte de um trabalho de **iniciação científica**.  
A implementação inclui o algoritmo de **Backpropagation** para treinamento e é comparada com pacotes estabelecidos do R.

---

## 🎯 Objetivos

- Implementar um MLP funcional com backpropagation  
- Testar em problemas clássicos (XOR e Iris)  
- Comparar performance com implementações existentes  
- Gerar análises gráficas de performance  

---

## 📁 Estrutura do Projeto
```
mlp-project/
├── mlp.R # Implementação principal do MLP
├── testXOR.R # Testes com dataset XOR
├── testIRIS.R # Testes com dataset Iris
├── compare.R # Comparação com outros pacotes
├── mlp_xor_comparison.pdf # Resultados XOR (gerado)
├── mlp_iris_comparison.pdf # Resultados Iris (gerado)
└── comparacao_implementacoes_fixed.pdf # Comparação (gerado)
```
---

## 🚀 Funcionalidades Implementadas

### 🔧 Funções Principais

- `mlp.create()` — Cria a arquitetura da rede neural  
- `mlp.forward()` — Propagação direta (forward propagation)  
- `mlp.train()` — Treinamento com backpropagation  
- `mlp.predict()` — Predição para novos dados  
- `mlp.accuracy()` — Cálculo de acurácia  
- `fnet()` / `dfnet()` — Função de ativação sigmoidal e sua derivada  

---

## 📊 Testes Realizados

### Problema XOR — [2, 4, 1]
- Problema não linearmente separável  
- Teste com múltiplas seeds (42, 123, 456, 789, 999)  
- Análise de convergência e acurácia  

### Dataset Iris — [4, 5, 3]
- Classificação multiclasse (3 espécies)  
- Normalização de features  
- Codificação one-hot para saída  

### Comparação com Pacotes do R
- **neuralnet** — Pacote tradicional para redes neurais  
- **RSNNS** — Stuttgart Neural Network Simulator  

---

## 📈 Resultados Destacados

### 🎯 Performance do XOR
- Nossa implementação: **100% de acurácia**  
- Convergência: ~2100 épocas (threshold 0.01)  
- Estabilidade: Consistente entre múltiplas seeds  

### 🌸 Performance do Iris
- Arquitetura: 4 entradas, 5 neurônios ocultos, 3 saídas  
- Acurácia: Resultados consistentes com diferentes inicializações  
- Normalização: Pré-processamento essencial para convergência  

---

## ⚡ Comparação com Outras Implementações

- Implementação própria mostra performance competitiva  
- Curvas de aprendizado consistentes  
- Código mais transparente e educacional  

---

## 🛠️ Como Usar

### Instalação das Dependências
```r
install.packages(c("ggplot2", "neuralnet", "RSNNS"))
Execução dos Testes
Teste com XOR
source("testXOR.R")

Teste com Iris
source("testIRIS.R")

Comparação com outros pacotes
source("compare.R")
