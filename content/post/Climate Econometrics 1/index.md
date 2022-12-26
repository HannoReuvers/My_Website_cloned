+++
title = "The greenhouse effect"
subtitle = "Understanding global warming"

date = 2018-06-01T00:00:00
lastmod = 2020-07-30T00:00:00
draft = false

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
  preview_only = true

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

## Short intro
We will first have to understand the origins of global warming before we can attempt to model the past and predict the future. Two key players are displayed in Figure 1. We see how both the global temperature and the global carbon dioxide[^1] ($\text{CO}_2$) concentration have been rising steadily over the last 40 years. The connection between these two graphs is commonly known as the "greenhouse effect".

{{< figure src="tempCO2.png" title="Figure 1: Yearly temperature anomalies and carbon dioxide concentrations for the time period 1980-2019. (a) The temperature anomaly is the temperature difference with respect to the 1951-1980 global average (source: https://climate.nasa.gov) . (b) Global $\text{CO}\_2$ concentrations as obtained from [NOAA Global Monitoring Laboratory](https://www.esrl.noaa.gov/gmd/ccgg/trends/global.html)." >}}

How exactly are these tiny $\text{CO}\_2$ molecules affecting the temperature on earth? The answer to this question is the topic of this post. We will look at two models: the bare-rock model and the layer model.[^2] Both models should be viewed as toy models as they describe a very much simplified version of reality. Indeed, the earth will be treated as a perfect sphere with the same temperature everywhere. The heat flows and spatial variation on our planet, e.g. ocean currents and altitude differences, are simply ignored. Notwithstanding their simplicity, these models will still provide all necessary ingredients to understand the origins of the greenhouse effect.

## The bare-rock model

The bare-rock model is a stylized model to explain the temperature on earth. This earth is represented by a large ball with a single uniform temperature. An energy balance between incoming sunlight and heat loss to space determines the temperature on our planet.

#### Energy received by the earth
When taking a walk on a beautiful summer day, you will feel the warmth of the sun. Similarly, the earth is continuously heated by the sun. How much energy is our spherical earth receiving from the sun? Let $I_{sun}$ denote the energy influx of sunlight, i.e. the amount of incoming energy per second and per $\text{m}^2$. To calculate the amount of energy absorbed per second, we have to multiply $I_{sun}$ by the area of the earth which is absorbing sunlight. An inspection of Figure 2 shows that the relevant area equals $\pi r_{earth}^2$. We additionally have to account for the fact that snow, ice, and clouds will reflect a fraction of the incoming sunlight back to space. This fraction is called the albedo and is generally denoted by $\alpha$. If a fraction $\alpha$ is reflected, then there remains a fraction $1-\alpha$ to be absorbed. Overall, the effective incoming flux of solar energy is:
\begin{equation}
\label{eq1}
F_{in} [\text{Watt}]= I_{sun} \times \pi r_{earth}^2 \times (1-\alpha).
\end{equation}
The units of this energy flux is Watt (also abbreviated by W).[^3]

{{< figure src="earthwithshadow.png" title="Figure 2: The earth will leave a circular shadow on a screen placed behind it. The area of this circle is $\pi r_{earth}^2$, where $r_{earth}^{}$ is the radius of our spherical planet." >}}

#### Emitted energy
The earth is not only absorbing energy, it is also emitting energy. The emitted energy is in the form of so-called black-body radiation. All warm objects emit this type of radiation and both the dominant wavelengths of the emitted light and the total energy flux depend on the body's temperature. Warmer objects emit radiation of shorter wavelength. This explains why we see a shining sun and a glow on the iron that has just been taken out of the fire. Our earth is much colder than either of these objects and is thus radiating light of much longer wavelength. These wavelenghts are too long for the human eye to observe.

Of particular interest for our bare-rock model is the energy flux from a black-body radiator. The amount of energy emitted per second and per $\text{m}^2$ is governed by the Stefan-Boltzmann law:
\begin{equation}
I_{black\text{ }body} = \sigma T^4,
\end{equation}
where $\sigma$ is the Stefan-Boltzmann constant (a constant of nature) and $T$ denotes the temperature in Kelvins.[^4] Each square meter of surface area can emit black-body radiation. Given that the total surface area of the earth is $4 \pi r_{Earth}^2$, the outgoing energy flux of our planet is
\begin{equation}
F_{out} [\text{Watt}] = \sigma T^4\times 4 \pi r_{earth}^2.
\end{equation}

#### Finding a balance...
Depending on the magnitudes of $F_{in}$ and $F_{out}$, we find ourselves in one of the following three situations:
1. $F_{in} > F_{out}$: The earth is receiving more energy than it is sending back to space. The accumulation of energy will cause the earth to heat up.
2. $F_{in} < F_{out}$: The nett heat flux is negative, that is, our spherical earth is steadily losing energy and hence cooling down.
3. $F_{in} = F_{out}$: The incoming and outgoing energy flux are exactly equal. There is neither heat gain nor heat loss and the temperature will remain constant.

If $F_{in} = F_{out}$, then our earth is in equilibrium -- or in steady-state -- and this provides a good indication of the earth's temperature according to our bare-rock model. Inserting the expressions for $F_{in}$ and $F_{out}$, this equilibrium temperature turns out to be:
$$
T_{equilibrium} = \left(  \frac{ (1-\alpha) I_{sun}}{4 \sigma} \right)^{1/4}.
$$
If we plug in some representative numbers,[^5] then this ball-shaped earth will have an equilibrium temperature of 254 K (or -19 $^\circ$C). This earth is too cold... we better add a blanket!

{{< figure src="layermodel.png" title="Figure 3: A schematic representation of the layer model. Sunlight shines unhindered through the layer and warms the earth. The radiation from the earth is trapped by the layer and re-emitted towards both the earth and the universe." >}}

## The layer model
Our blanket is an additional layer placed between the earth and the sun, see Figure 3. This layer is special. The incoming light from the sun will pass straight through it, but the outgoing radiation from the earth is absorbed and subsequently re-emitted. Our new task is to determine the resulting equilibrium temperature of the earth. We again match the incoming and outgoing energy fluxes but now we should do so for the earth and the layer simulaneously. Additionally, we set the sun's energy flux per unit of area equal to $I_{sunlight}= \frac{1}{4} (1-\alpha) I_{sun}$.[^6] The equilibrium equations are 
$$
 I_{sunlight} + I_{layer,down} = I_{earth} \iff \frac{1}{4} (1-\alpha) I_{sun} + \sigma T_{layer}^4 = \sigma T_{earth}^4
$$
and
$$
 I_{earth} = I_{layer,up} + I_{layer,down} \iff \sigma T_{earth}^4 =  \sigma T_{layer}^4 +  \sigma T_{layer}^4
$$
for the earth and layer, respectively. We can solve these two equations and find the earth's temperature in this layer model:
$$
 T_{earth} =  \left(  \frac{ 2 (1-\alpha) I_{sun}}{4 \sigma} \right)^{1/4}.
$$
A quick comparison with $T_{equilibrium}$ shows that the new temperature $T\_{earth}$ is a factor $2^{1/4}\approx 1.1892$ higher. The new temperature on earth is now 302 K (or about 29 $^\circ$C). The current temperature is admittedly a bit too high but we will also see that $\text{CO}\_2$ is not behaving as extreme as our layer. That is, this greenhouse gas does not absorb all the radiation coming from earth. Most importantly, we can now understand why the additional layer works as a blanket. It creates an additional energy flow $I\_{layer,down}$ back to earth that causes the earth to warm up. 

## The analogy between carbon dioxide and the layer
Carbon dioxide behaves similarly to our layer. The incoming sunlight from our warm sun has a relatively short wavelenght and this light will not (or hardly) interact with the $\text{CO}\_2$ molecules in the air. As the earth is much colder than the sun, it will emit black body radiation with longer wavelength. This radiation is partly(!) absorbed by the carbon dioxide molecules in the air and released into any arbitrary direction. This process creates an energy flow back to earth much the same as $I\_{layer,down}$ in Figure 3. The higher the $\text{CO}\_2$ concentration in the atmosphere, the more energy is channeled back to earth, and the more global warming. Finally, do note that there is nothing special about carbon dioxide. The warming mechanism behind the other greenhouse gasses is the same!

## Notes
[^1]: Of course, there are also other greenhouse gasses (e.g. water vapor, methane, nitrous oxide, and ozone). Feel free to read "greenhouse gasses" instead of "carbon dioxide" throughout this post.
[^2]: The discussions of the bare-rock and layer model are inspired by chapters 2-4 from <i>Global Warming: Understanding the Forecast</i> by [David Archer](https://geosci.uchicago.edu/people/david-archer/).
[^3]: 1 Watt is equal to 1 Joule per second.
[^4]: It is easy to convert a temperature from degree Celsius to Kelvin, just add 273.15. For example, water freezes at 0 $^\circ$C or equivalently 273.15 K.
[^5]: The numerical value of the Stefan-Boltzmann constant is $\sigma = 5.670\times 10^{-8} \text{ J} \text{ s}^{-1} \text{ m}^{-2} \text{ K}^{-4}$. For $\alpha$ and $I_{sun}$, we follow the book by David Archer, and use $\alpha=0.3$ and $I_{sun}= 1350 \text{ W m}^{-2}$.
[^6]: The factor $(1-\alpha)$ accounts for the earth's albedo. Subsequently recall that: (1) a surface area of $\pi r_{earth}^2$ is facing the sun and thus absorbing energy, and (2) that the total surface area of the earth is $4 \pi r_{earth}^2$. The effective energy influx <i>per unit of surface area</i> thus requires a factor $\frac{1}{4}$.
