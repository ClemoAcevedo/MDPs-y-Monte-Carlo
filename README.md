# IIC3675 - Tarea 2: MDPs y Monte Carlo

Este repositorio contiene el código para resolver Procesos de Decisión de Markov (MDPs) utilizando métodos de Programación Dinámica y Monte Carlo, como parte de la Tarea 2 del curso **IIC3675**.

---

## Descripción del Proyecto

El repositorio está organizado en dos carpetas principales: **MDPs** y **MonteCarlo**.

### MDPs (Programación Dinámica)
Contiene las implementaciones de **Iterative Policy Evaluation** y **Value Iteration** para los problemas `GridProblem`, `CookieProblem` y `GamblerProblem`.  
Los experimentos analizan:

- El valor de políticas aleatorias  
- El efecto del factor de descuento (γ)  
- La optimalidad de las políticas greedy resultantes  

### MonteCarlo (MC)
Contiene la implementación de **On-policy every-visit MC control** para entrenar agentes en los entornos `BlackjackEnv` y `CliffEnv`.  
Se realizan múltiples corridas para analizar estabilidad y rendimiento a lo largo del entrenamiento.

---

## Estructura de Archivos

```
├── MDPs
│   ├── Main.py             # Experimentos de Programación Dinámica
│   ├── algorithms.py       # Implementaciones de algoritmos de DP
│   └── Problems/           # Definiciones de los MDPs
│
├── MonteCarlo
│   ├── Main.py             # Experimentos de Monte Carlo
│   ├── MonteCarlo.py       # Implementación del agente MC
│   └── Environments/       # Definiciones de los ambientes MC
```

Los gráficos de **MDPs** se generan dentro de la carpeta `MDPs/`.  
Los gráficos y trayectorias de **MonteCarlo** se generan dentro de la carpeta `MonteCarlo/`.

---

## Prerrequisitos

Se requiere **Python 3** con las siguientes librerías:

```bash
pip install numpy matplotlib
```

---

## Cómo Replicar los Experimentos

### 1. Programación Dinámica (MDPs)

Ejecuta desde la carpeta `MDPs`:

```bash
python Main.py
```

**Salida esperada:**  
- En consola:  
  - Valores de estado inicial para políticas aleatorias y greedy  
  - Impacto de γ en convergencia  
  - Resultados de Value Iteration y políticas óptimas del GamblerProblem  
- Archivos generados (en `MDPs/`):  
  - `gambler_policy_ph_0.25.png`  
  - `gambler_policy_ph_0.4.png`  
  - `gambler_policy_ph_0.55.png`  

---

### 2. Monte Carlo (MC)

Ejecuta desde la carpeta `MonteCarlo`:

```bash
python Main.py
```

**Salida esperada:**  
- En consola:  
  - Progreso detallado de entrenamiento (5 corridas)  
  - Políticas finales aprendidas (Blackjack)  
  - Trayectorias finales (Cliff Walking)  

⚠️ Nota: La sección de **Cliff Walking con ancho 12** puede demorar demasiado en entrenar.  
Se recomienda **cortar la ejecución** y justificar esta decisión en el informe, ya que forma parte de las consideraciones esperadas.

**Archivos generados (en `MonteCarlo/`, con timestamp `YYYYMMDD_HHMM`):**  
  - `blackjack_perf_YYYYMMDD_HHMM.png`  
  - `cliff6_perf_YYYYMMDD_HHMM.png`  
  - `cliff6_perf_YYYYMMDD_HHMM_zoomed.png`  
  - `cliff12_perf_YYYYMMDD_HHMM.png`  
  - `cliff12_perf_YYYYMMDD_HHMM_zoomed.png`  
  - `cliff6_YYYYMMDD_HHMM_final_trajectory_runXX.png`  
  - `cliff12_YYYYMMDD_HHMM_final_trajectory_runXX.png`  

---

## Autor
Tarea 2 - Curso **IIC3675: Reinforcement Learning & Recommender Systems**
