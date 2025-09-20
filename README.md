# 🌍 Life Expectancy-Based Segmentation of Countries

This project clusters countries based on life expectancy data using unsupervised learning techniques. By applying dimensionality reduction and clustering algorithms, we uncover meaningful groupings that reflect global health and development patterns.

## 📦 Dataset

- `life_expectancy.csv`: Contains life expectancy metrics across countries.

## 🧪 Methodology

1. **Preprocessing**
   - Missing values handled appropriately.
   - Features standardized using `StandardScaler`.

2. **Dimensionality Reduction**
   - Applied `PCA` to reduce dimensionality while preserving variance.

3. **Clustering**
   - Used `DBSCAN` to identify clusters of countries with similar life expectancy profiles.
   - DBSCAN is ideal for detecting clusters of arbitrary shape and handling noise.

4. **Evaluation**
   - Used `Silhouette Score` to evaluate clustering quality and tune DBSCAN parameters (`eps`, `min_samples`).

## 📈 Results

- Countries grouped into distinct clusters based on life expectancy trends.
- PCA visualization revealed clear separations among clusters.
- Silhouette analysis guided optimal DBSCAN configuration.

## 🛠️ Technologies Used

- Python
- NumPy, Pandas
- Scikit-learn
- Matplotlib, Seaborn

## 📁 Repository Structure

├── life_expectancy.csv # Dataset 
├── Life_Expectancy.py # Main script for preprocessing, PCA, DBSCAN, and visualization 
├── README.md # Project documentation


## 📜 License

This project is licensed under the MIT License.
