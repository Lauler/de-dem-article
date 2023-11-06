library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)

df <- arrow::read_parquet("data/results/all_results_preprocessed.parquet")

# Select 1 observation per speaker
# We have previously calculated summary stats for each speaker in variables such as
# author_*_mistake, author_*_correct, author_comments_per_year
authors <- df %>%
  filter(maincorpus %in% c("familjeliv", "reddit")) %>%
  group_by(speaker) %>% 
  filter(row_number()==1 & !is.na(speaker))

authors <- authors %>%
  filter(author_comments_total >= 10)

authors <- authors %>%
  select(speaker, author_de_mistake, author_dem_mistake, author_all_mistake, author_det_mistake)

authors <- authors %>%
  pivot_longer(cols = starts_with("author"),
               names_to = "mistake_type",
               values_to = "nr_mistakes")


authors <- authors %>%
  group_by(mistake_type) %>%
  arrange(-nr_mistakes) %>%
  mutate(cum_sum = cumsum(nr_mistakes),
         cdf = cum_sum/sum(nr_mistakes),
         author_fraction = row_number()/n())



p <- ggplot(data=authors %>% filter(mistake_type != "author_all_mistake"), aes(x=author_fraction, y=cdf, colour=mistake_type)) +
  geom_line(aes(linetype = mistake_type)) +
  theme_minimal(base_size=10) +
  scale_x_continuous(labels = scales::percent, 
                     breaks = scales::pretty_breaks(10)) +
  scale_y_continuous(labels = scales::percent, limits = c(0, NA), breaks = scales::pretty_breaks(6)) +
  scale_color_hue(labels = c("De (objekt)", "Dem (subj/det)", "De(t) (singular)")) +
  scale_linetype_manual(labels = c("De (objekt)", "Dem (subj/det)", "De(t) (singular)"), values = 1:3) +
  theme(plot.title = element_text(size=11),
        panel.grid.major.y = element_line(colour="grey62", linewidth=0.3, linetype=2),
        panel.grid.major.x = element_line(colour="grey62", linewidth=0.1, linetype=1),
        panel.grid.minor.x = element_blank(),
        text=element_text(family="Palatino"),
        plot.subtitle=element_text(size=5.5),
        legend.key.width = unit(0.6, "cm"),
        legend.position = "right") +
  labs(title = "Andel av samtliga förväxlingar författade av en given andel skribenter",
       x = "Andel skribenter",
       y = "Kumulativ andel förväxlingar",
       color = "Typ av förväxling",
       linetype = "Typ av förväxling",
       caption = "Korpusar: Familjeliv & Reddit",
       subtitle = "Resultat presenteras för skribenter som använt de och dem 10 eller fler gånger (sammanlagt)")



ggsave("plots/skribenter_cdf/kumulativa_forvaxlingar_skribent.png", 
       plot = p, width=1900, height=1000, units="px", dpi=300, bg = "white")
