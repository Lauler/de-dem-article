library(ggplot2)
library(dplyr)

df <- list(Bloggmix = readr::read_tsv("data/de(m)_vs_dom/bloggmix.tsv"), 
           GP = readr::read_tsv("data/de(m)_vs_dom/gp.tsv"), 
           SVT = readr::read_tsv("data/de(m)_vs_dom/svt.tsv"))

df <- bind_rows(df, .id = "corpus") %>%
  select(-v1ipm, -v2ipm) %>%
  mutate(post_year = lubridate::ymd(period, truncated=2L))

df_familjeliv <- readr::read_tsv("data/de(m)_vs_dom/familjeliv_sentence_10000_firstage18_de(m)_vs_dom_t0.tsv")

df_familjeliv <- df_familjeliv %>%
  mutate(birth_year = period - age) %>%
  filter(birth_year != 1970) # filter out 1970

#### By Year ####
df_familjeliv_year <- df_familjeliv %>%
  group_by(period) %>%
  summarise(total = sum(total),
            v1abs = sum(v1abs),
            v2abs = sum(v2abs)) %>%
  mutate(corpus = "Familjeliv",
         v1rel = v1abs / total,
         v2rel = v2abs / total,
         post_year = lubridate::ymd(period, truncated=2L))

df <- bind_rows(df, df_familjeliv_year)

# Confidence interval bounds using normal approx of binomial: p +- z * sqrt(p(1-p)/n)
alpha <- 0.05 # 95% conf interval
df$conf_lower <- df$v2rel - qnorm(p = 1 - alpha/2) * sqrt((df$v2rel * (1 - df$v2rel)) / df$total)
df$conf_upper <- df$v2rel + qnorm(p = 1 - alpha/2) * sqrt((df$v2rel * (1 - df$v2rel)) / df$total)

df$conf_lower[df$conf_lower < 0] <- 0 # Set negative lower conf bound values to 0
df$conf_upper[df$conf_upper > 1] <- 1

p1 <- ggplot(data = df %>% filter(total > 500), aes(x = post_year, y = v2rel)) +
  geom_line(aes(linetype=corpus, color = corpus), linewidth = 0.6) +
  geom_point(aes(fill=corpus, shape=corpus, color = corpus), colour="black", stroke=0.3) +
  geom_ribbon(aes(ymin = conf_lower, ymax = conf_upper, fill = corpus), alpha = 0.25, show.legend=FALSE) +
  theme_minimal(base_size=10) +
  scale_shape_manual(values = 21:25) +
  scale_x_date(breaks = scales::pretty_breaks(16),
               guide = guide_axis(n.dodge=2)) +
  scale_y_continuous(labels = scales::percent, limits = c(0, 0.26), breaks = scales::pretty_breaks(6)) +
  theme(plot.title = element_text(size=11),
        panel.grid.major.y = element_line(colour="grey62", linewidth=0.3, linetype=2),
        panel.grid.major.x = element_line(colour="grey62", linewidth=0.1, linetype=1),
        panel.grid.minor.x = element_blank(),
        text=element_text(family="Palatino"),
        plot.subtitle=element_text(size=5.5),
        legend.key.width = unit(0.6, "cm")) +
  guides(linetype = guide_legend(override.aes = list(size = 1.5))) +
  labs(x = "År",
       y = 'Andel dom',
       # title = 'Andel "dom" av totala antalet användningar av "de/dem/dom" per korpus',
       color = "Korpus",
       shape = "Korpus",
       linetype = "Korpus",
       # size = "Antal obs.",
       fill = "Korpus")

ggsave("plots/dom_vs_dedemdom/dom_vs_demdemdom_by_corpus.png", 
       plot = p1, width=1900, height=1000, units="px", dpi=300, bg = "white")


p2 <- ggplot(data = df %>% filter(total > 500), aes(x = post_year, y = v2rel, color = corpus)) +
  geom_line(aes(linetype=corpus), linewidth = 0.8) +
  geom_point(aes(fill=corpus, shape=corpus, size=total), colour="black", stroke=0.5) +
  theme_minimal(base_size=19) +
  scale_shape_manual(values = 21:25) +
  scale_x_date(breaks = scales::pretty_breaks(16),
               guide = guide_axis(n.dodge=2)) +
  scale_y_continuous(labels = scales::percent, limits = c(0, 0.26), breaks = scales::pretty_breaks(6)) +
  scale_size_continuous(labels = function(x) format(x, big.mark = " ", decimal.mark = ".", scientific = FALSE),
                        limits = c(1000, 500000),
                        breaks = c(1200, 3000, 5000, 10000, 50000, 100000, 500000),
                        trans = "log",
  ) +
  theme(plot.title = element_text(size=11),
        panel.grid.major.y = element_line(colour="grey62", linewidth=0.5, linetype=2),
        panel.grid.major.x = element_line(colour="grey62", linewidth=0.2, linetype=1),
        panel.grid.minor.x = element_blank(),
        text=element_text(family="Palatino"),
        plot.subtitle=element_text(size=5.5),
        legend.key.width = unit(1.2, "cm"),
        legend.key.height = unit(0.8, "cm")) +
  guides(linetype = guide_legend(override.aes = list(size = 2.5), order=1),
         shape = guide_legend(order=1),
         fill = guide_legend(order=1),
         color = guide_legend(order=1),
         size = guide_legend(order=2, override.aes = list(color = "grey15"))) +
  labs(x = "År",
       y = 'Andel dom',
       # title = 'Andel "dom" av totala antalet användningar av "de/dem/dom" per korpus',
       color = "Korpus",
       shape = "Korpus",
       linetype = "Korpus",
       size = "Antal obs.",
       fill = "Korpus")

ggsave("plots/dom_vs_dedemdom/dom_vs_demdemdom_by_corpus_pointsize.png", 
       plot = p2, width=1900, height=1000, units="px", dpi=150, bg = "white")



#### By Generations ####
df_familjeliv_gen <- df_familjeliv %>%
  # filter(total >= 10) %>% # 10 or more comments per author
  group_by(period, agebin) %>%
  summarise(total = sum(total),
            v1abs = sum(v1abs),
            v2abs = sum(v2abs),
            n_authors = unique(username) %>% length()) %>%
  mutate(corpus = "Familjeliv",
         v1rel = v1abs / total,
         v2rel = v2abs / total,
         post_year = lubridate::ymd(period, truncated=2L),
         generation = recode(agebin, Gen1 = "Före 1970", Gen2 = "1970-1979", Gen3 = "1980-1989", Gen4 = "Från 1990"),
         generation = factor(generation, levels = c("Från 1990", "1980-1989", "1970-1979", "Före 1970")),
         conf_lower = v2rel - qnorm(p = 1 - alpha/2) * sqrt((v2rel * (1 - v2rel)) / total),
         conf_upper = v2rel + qnorm(p = 1 - alpha/2) * sqrt((v2rel * (1 - v2rel)) / total),
         conf_lower = ifelse(test = conf_lower < 0, yes = 0, no = conf_lower),
         conf_upper = ifelse(test = conf_lower > 1, yes = 1, no = conf_upper)) %>%
  filter(n_authors >= 20) %>%
  ungroup()

p3 <- ggplot(data = df_familjeliv_gen %>% filter(total > 500), aes(x = post_year, y = v2rel)) +
  geom_line(aes(linetype=generation, color = generation), linewidth = 0.6) +
  geom_point(aes(fill=generation, shape=generation, color = generation), colour="black", size=1.4, stroke=0.3) +
  geom_ribbon(aes(ymin = conf_lower, ymax = conf_upper, fill = generation), alpha = 0.25, show.legend=FALSE) +
  theme_minimal(base_size=10) +
  scale_shape_manual(values = 21:24) +
  scale_x_date(breaks = scales::pretty_breaks(16),
               guide = guide_axis(n.dodge=2)) +
  scale_y_continuous(labels = scales::percent, limits = c(0, NA), breaks = scales::pretty_breaks(6)) +
  theme(plot.title = element_text(size=11),
        panel.grid.major.y = element_line(colour="grey62", linewidth=0.3, linetype=2),
        panel.grid.major.x = element_line(colour="grey62", linewidth=0.1, linetype=1),
        panel.grid.minor.x = element_blank(),
        text=element_text(family="Palatino"),
        plot.subtitle=element_text(size=5.5),
        legend.key.width = unit(0.6, "cm")) +
  guides(linetype = guide_legend(override.aes = list(size = 1.5))) +
  labs(x = "År",
       y = 'Andel dom',
       # title = 'Andel "dom" av totala antalet användningar av "de/dem/dom" över generationer',
       color = "Generation",
       shape = "Generation",
       linetype = "Generation",
       fill = "Generation")
df_familjeliv_gen
ggsave("plots/dom_vs_dedemdom/dom_vs_demdemdom_by_generation.png", 
       plot = p3, width=1900, height=1000, units="px", dpi=300, bg = "white")


p4 <- ggplot(data = df_familjeliv_gen %>% filter(total > 500), aes(x = post_year, y = v2rel, color = generation)) +
  geom_line(aes(linetype=generation), linewidth = 0.9) +
  geom_point(aes(fill=generation, shape=generation, size=total), colour="black", stroke=0.5) +
  # geom_point(aes(fill=generation, shape=generation), colour="black", size=1.4, stroke=0.3) +
  theme_minimal(base_size=19) +
  scale_shape_manual(values = 21:24) +
  scale_x_date(breaks = scales::pretty_breaks(16),
               guide = guide_axis(n.dodge=2)) +
  scale_y_continuous(labels = scales::percent, limits = c(0, NA), breaks = scales::pretty_breaks(6)) +
  scale_size_continuous(labels = function(x) format(x, big.mark = " ", decimal.mark = ".", scientific = FALSE),
                        limits = c(400, 250000),
                        breaks = c(500, 1000, 5000, 10000, 50000, 100000, 250000),
                        trans = "log",
  ) +
  theme(plot.title = element_text(size=11),
        panel.grid.major.y = element_line(colour="grey62", linewidth=0.5, linetype=2),
        panel.grid.major.x = element_line(colour="grey62", linewidth=0.2, linetype=1),
        panel.grid.minor.x = element_blank(),
        text=element_text(family="Palatino"),
        plot.subtitle=element_text(size=5.5),
        legend.key.width = unit(1.2, "cm"),
        legend.key.height = unit(0.8, "cm")) +
  guides(linetype = guide_legend(override.aes = list(size = 2.5), order=1),
         shape = guide_legend(order=1),
         fill = guide_legend(order=1),
         color = guide_legend(order=1),
         size = guide_legend(order=2, override.aes = list(color = "grey15"))) +
  labs(x = "År",
       y = 'Andel dom',
       # title = 'Andel "dom" av totala antalet användningar av "de/dem/dom" över generationer',
       size = "Antal obs.",
       color = "Generation",
       shape = "Generation",
       linetype = "Generation",
       fill = "Generation")

ggsave("plots/dom_vs_dedemdom/dom_vs_demdemdom_by_generation_pointsize.png", 
       plot = p4, width=1900, height=1000, units="px", dpi=150, bg = "white")
