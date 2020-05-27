 function capacity = aed_capacity(x, pdf, snr, number_receivers, step)
    probabilities = pdf.*step;
    capacity = number_receivers.*sum(log2(1 + snr.*x).*probabilities);
 end
