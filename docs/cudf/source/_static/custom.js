// Copyright (c) 2022, NVIDIA CORPORATION.

function update_switch_theme_button() {
    current_theme = document.documentElement.dataset.mode;
    if (current_theme == "light") {
        document.getElementById("theme-switch").title = "Switch to auto theme";
    } else if (current_theme == "auto") {
        document.getElementById("theme-switch").title = "Switch to dark theme";
    } else {
        document.getElementById("theme-switch").title = "Switch to light theme";
    }
}

$(document).ready(function() {
    var observer = new MutationObserver(function(mutations) {
        update_switch_theme_button();
    })
    observer.observe(document.documentElement, {
        attributes: true,
        attributeFilter: ['data-theme']
    });
});
