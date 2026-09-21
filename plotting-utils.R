COLORPAL <- c(orange = '#D55E00',
              blue = '#0072B2',
              lightblue = '#56B4E9',
              green = '#009E73',
              magenta = '#CC79A7',
              black = '#000000')

plottheme <- function(base_size = 12, base_family = '', dark = FALSE, bg = FALSE) {
  library(ggplot2)
  gr <- gray(seq(0, 1, length.out = 11))
  if (isTRUE(dark)) {
    gr <- rev(gr)
  }

  bg <- if (isTRUE(bg)) {
    gr[[11]]
  } else if (isFALSE(bg)) {
    'transparent'
  } else {
    bg
  }

  theme_bw(base_size = base_size, base_family = base_family) +
    theme(plot.title = element_text(size = rel(0.9),
                                    hjust = 0.5),
          plot.margin = margin(0.2, 0.4, 0.5, 0.2, 'lines'),
          panel.background = element_rect(fill = bg, color = NA),
          plot.background = element_rect(fill = bg, color = NA),
          legend.title = element_text(size = rel(0.7)),
          legend.text = element_text(size = rel(0.7)),
          axis.title = element_text(size = rel(0.75)),
          axis.text = element_text(size = rel(0.7)),
          axis.title.y = element_text(vjust = 1),
          axis.title.x = element_text(vjust = 0),
          # panel.grid.major = element_blank(),
          panel.grid.major = element_line(
            color = gr[[7]], linewidth = rel(0.5), linetype='dotted'),
          panel.grid.minor = element_blank(),
          strip.background = element_rect(
            fill = gr[[11]], color = gr[[4]], linewidth = 0.3),
          strip.text = element_text(size = rel(0.75), color = gr[[1]]),
          panel.border = element_rect(color = gr[[4]], linewidth = 0.3),
          # panel.border = element_blank(),
          text = element_text(color = gr[[1]]),
          # axis.line.x.bottom = element_line(color = gr[[10]], linewidth = 0.5),
          # axis.line.y.left = element_line(color = gr[[2]], linewidth = 0.5),
          axis.text.x = element_text(color = gr[[2]]),
          axis.text.y = element_text(color = gr[[2]]),
          legend.background = element_blank())
}
