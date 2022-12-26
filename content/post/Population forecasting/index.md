+++
title = "Population forecasting with the Lee-Carter model"
subtitle = "Something"

date = 2021-04-03T00:00:00
lastmod = 2021-04-03T00:00:00
draft = true

# Authors. Comma separated list, e.g. `["Bob Smith", "David Jones"]`.
authors = []

tags = []
summary = "Understanding global warming"

# Link to related code
# url_code = "https://github.com/HannoReuvers/Software_comparison_DW"

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["deep-learning"]` references 
#   `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
# projects = ["internal-project"]

# Featured image
# To use, add an image named `featured.jpg/png` to your project's folder. 
[image]
  # Caption (optional)
  # caption = "Image credit: [**Unsplash**](https://unsplash.com/photos/CpkOjOcXdUY)"

  # Focal point (optional)
  # Options: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight
  focal_point = ""

# Show image only in page previews?
  preview_only = false

# Set captions for image gallery.

[[gallery_item]]
album = "gallery"
image = "theme-default.png"
caption = "Default"

[[gallery_item]]
album = "gallery"
image = "theme-ocean.png"
caption = "Ocean"

[[gallery_item]]
album = "gallery"
image = "theme-forest.png"
caption = "Forest"

[[gallery_item]]
album = "gallery"
image = "theme-dark.png"
caption = "Dark"

[[gallery_item]]
album = "gallery"
image = "theme-apogee.png"
caption = "Apogee"

[[gallery_item]]
album = "gallery"
image = "theme-1950s.png"
caption = "1950s"

[[gallery_item]]
album = "gallery"
image = "theme-coffee-playfair.png"
caption = "Coffee theme with Playfair font"

[[gallery_item]]
album = "gallery"
image = "theme-cupcake.png"
caption = "Cupcake"
+++

## Introducing the 
The population of a country consists of people who each have their unique features and talents. As a simpliflication, we focus on a single individual for now. That is, we let $(x)$ denote a life aged $x$ (where $x\geq 0$). The death of $(x)$ is unknown and we therefore introduce the continuous random variable $T_x$ to model the remaining lifetime. This implies that $(x)$ will live to the age of $x+T_x$. Further notation related to the random variable $T_x$ is collected in Table 1.


| Name | Notation | ``Probability that $(x)$...''
|---|---|---|---|
| Distribution function | $F_x(t) = \mathbb{P}(T_x\leq t)$ $[ _tq_x ]$    | ... dies before age $x+t$
| Survival function | $S_x(t) = 1- F_x(t)$ $[ _tp_x ]$ | ... survives to at least age $x+t$
|                            |
 1. Explain force of mortality (use newly downloaded book)
 2. Lee-Carter model
 3. Estimation
 
 The remainder of this post consists of four parts. I first illustrate/explain the two advantages mentioned above. The subsequent discussion revolves around the choice of the penalization parameter. Finally, there is a section on Monte Carlo simulation. This 

## The bare-rock model

The bare-rock model is a stylized model to explain the temperature on earth. This earth is represented by a large ball with a single uniform temperature. An energy balance between incoming sunlight and heat loss to space determines the temperature on our planet.

#### Energy received by the earth
When taking a walk on a beautiful summer day, you will feel the warmth of the sun. Similarly, the earth is continuously heated by the sun. How much energy is our spherical earth receiving from the sun? Let $I_{sun}$ denote the energy influx of sunlight, i.e. the amount of incoming energy per second and per $\text{m}^2$. To calculate the amount of energy absorbed per second, we have to multiply $I_{sun}$ by the area of the earth which is absorbing sunlight. An inspection of Figure 2 shows that the relevant area equals $\pi r_{earth}^2$. We additionally have to account for the fact that snow, ice, and clouds will reflect a fraction of the incoming sunlight back to space. This fraction is called the albedo and is generally denoted by $\alpha$. If a fraction $\alpha$ is reflected, then there remains a fraction $1-\alpha$ to be absorbed. Overall, the effective incoming flux of solar energy is:
\begin{equation}
\label{eq1}
F_{in} [\text{Watt}]= I_{sun} \times \pi r_{earth}^2 \times (1-\alpha).
\end{equation}
The units of this energy flux is Watt (also abbreviated by W).[^3]

## The simulation setting
The data generating process for the simulations is inspired by Example 4.1 of Fan and Li (2001).[^1] That is, we consider the linear regression model
$$
 y_i = \mathbf x_i^\prime \boldsymbol \beta + \varepsilon_i, \qquad\qquad i=1,\ldots,n,
$$
where $\boldsymbol \beta = [3,1.5,0,0,2,0,0,0]^\prime$ and the error terms $\\{ \varepsilon_i \\}_{i=1}^n$ are drawn as independent standard normally distributed random variables. In each Monte Carlo replication, the vector of explanatory variables $\mathbf x_i$ is redrawn from a multivariate normal distribution with mean vector $\boldsymbol 0$ and a Toeplitz covariance matrix, i.e.
$$
 \mathbf x_i \sim \mathcal{N}\left( \boldsymbol 0 , \begin{bmatrix} 1 & \rho & \cdots & \rho^7 \\\ \rho &1 & \cdots &\rho^6 \\\ \vdots & \vdots & \ddots & \vdots \\\ \rho^7 & \rho^6 & \cdots & 1\end{bmatrix} \right).
$$

## Notes
[^1]: J. Fan and R. Li (2001), _Variable Selection via Nonconcave Penalized Likelihood and its Oracle Properties_, Journal of the American Statistical Association
