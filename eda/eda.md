# **Dataset Distribution Analysis (Using All Three Plots)**

### **1. Country Frequency Distribution**

The bar chart reveals that the GeoGuessr dataset follows a **power-law distribution**: a handful of countries contain thousands of images (e.g., United States, Japan, France, United Kingdom), while the majority contain only a few dozen, and in many cases, fewer than ten. Several countries have **only a single image**, making them effectively impossible to learn.
This immediately signals a **severe long-tail imbalance**.

---

### **2. Cumulative Coverage Curve**

The cumulative distribution plot quantifies the imbalance more precisely.
Sorting countries by frequency and accumulating their contributions shows:

* The **top ~23 countries account for 90% of the entire dataset**.
* The remaining ~100 countries combined account for only 10%.
* The curve rises sharply at the beginning and flattens quickly, characteristic of a **heavy-tailed dataset**.

This confirms that the dataset is not uniformly spread across 120+ classes; rather, it behaves like a dataset with **~20 meaningful classes and 100 ultra-sparse ones**.

---

### **3. Visual Difficulty Showcase**

Sampling images across high-frequency, mid-frequency, and extremely low-frequency countries highlights the practical impact of this imbalance:

* Countries with thousands of images show huge visual diversity in roads, landscapes, vegetation, and infrastructure.
* Mid-frequency countries show more consistent features but still enough variation for a model to learn from.
* Countries with 1–5 images provide *no meaningful learning signal*, the model effectively has to “memorize” isolated samples, which it cannot generalize from.

This visualization reinforces that many countries are **not only rare but also visually ambiguous**, making classification even harder.

---

# **Unified Interpretation (What These Plots Together Mean)**

Taken together, these plots reveal that:

1. **The dataset is dominated by a small number of countries.**
   A classifier trained naively on this dataset will learn to predict these countries overwhelmingly often.

2. **Most countries lack enough examples for meaningful learning.**
   The model will almost certainly fail on 80–100 of the classes simply because the model never sees enough examples to extract patterns.

3. **Training a 120-way classifier is mathematically possible but practically flawed.**
   Even with perfect architecture and tuning, the model will default to the top classes since they dominate gradient updates.

4. **Dataset balancing or class reduction is required** to build a truly functional model.

---

# **Final Plotting Summary Paragraph (paste directly into your report)**

> **Our dataset analysis shows a highly skewed class distribution where the top ~23 countries comprise 90% of all images, while over 40 countries contain fewer than 10 images and several contain only one. The cumulative coverage curve highlights the heavy-tailed structure of the dataset, and the visual difficulty showcase confirms that many of the rare countries offer too little data for the model to learn any reliable features. Together, these plots indicate that the dataset is effectively learnable only for the top 20–30 countries, and that without balancing, the model will overwhelmingly predict the high-frequency classes while ignoring the long tail. This imbalance must be addressed through class reduction, resampling, or weighted loss functions to ensure meaningful training.**