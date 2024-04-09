import beta from 'https://cdn.jsdelivr.net/gh/stdlib-js/stats-base-dists-beta@esm/index.mjs';
import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7/+esm";


function plotBetaPdf(plotSelector, alpha, beta){
    let viz = document.querySelector(plotSelector);
    let sliderAlpha = document.querySelector(plotSelector + "~ input.alpha");
    let sliderBeta = document.querySelector(plotSelector + "~ input.beta");

    const width = 640;
    const height = 300;
    const marginTop = 20;
    const marginRight = 20;
    const marginBottom = 30;
    const marginLeft = 40;

    let svg = d3.create("svg")
                .attr("width", width)
                .attr("height", height);
    
    let xScale = d3.scaleLinear()
                    .domain([0, 1])
                    .range([marginLeft, width - marginRight]);
    
    let yScale = d3.scaleLinear()
                    .domain([0, 20])
                    .range([height - marginBottom, marginTop]);
    
    svg.append("g")
        .attr("transform", `translate(0,${height - marginBottom})`)
        .call(d3.axisBottom(xScale));
        
    svg.append("g")
        .attr("transform", `translate(${marginLeft},0)`)
        .call(d3.axisLeft(yScale));

    viz.append(svg.node());
}

plotBetaPdf("#viz-beliefs");
