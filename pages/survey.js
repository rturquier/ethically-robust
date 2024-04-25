import { pdf as betaPdf } from 'https://cdn.jsdelivr.net/gh/stdlib-js/stats-base-dists-beta@esm/index.mjs';
import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7/+esm";

/**
 * Convert mean and variance of beta distribution to parameters of beta 
 * distribution.
 * See https://en.wikipedia.org/wiki/Beta_distribution.
 */
function momentsToParameters(mean, variance){
    const commonFactor = (mean * (1 - mean)) / variance - 1;
    const alpha = commonFactor * mean;
    const beta = commonFactor * (1 - mean);
    return [alpha, beta];
}


function plotBetaPdf(plotSelector){
    const viz = document.querySelector(plotSelector);
    const sliderMean = document.querySelector(plotSelector + "~ input.mean");
    const sliderSigma = document.querySelector(plotSelector + "~ input.sigma");

    const width = viz.offsetWidth;
    const height = width / 2.5;
    const marginTop = 20;
    const marginRight = 20;
    const marginBottom = 30;
    const marginLeft = 40;
    const xPrecision = 0.001;

    const mean = sliderMean.valueAsNumber;
    const sigma = sliderSigma.valueAsNumber;
    const variance = sigma**2;
    const [alpha, beta] = momentsToParameters(mean, variance);

    if (d3.select("svg") != false){
        d3.select("svg").remove();
    }

    let svg = d3.create("svg")
                .attr("width", width)
                .attr("height", height);
    
    let xScale = d3.scaleLinear()
                    .domain([0, 1])
                    .range([marginLeft, width - marginRight]);
    
    let yScale = d3.scaleLinear()
                    .domain([0, 5])
                    .range([height - marginBottom, marginTop]);
    
    svg.append("g")
        .attr("transform", `translate(0,${height - marginBottom})`)
        .call(d3.axisBottom(xScale));
        
    svg.append("g")
        .attr("transform", `translate(${marginLeft},0)`)
        .call(d3.axisLeft(yScale).tickFormat("").tickValues([1]).tickSizeOuter(0));

    const points = [];
    for (let x = 0 + xPrecision; x <= 1 - xPrecision; x += xPrecision){
        let y = betaPdf(x, alpha, beta);
        points.push([x, y]);
    }

    let line = d3.line()
                .x(d => xScale(d[0]))
                .y(d => yScale(d[1]));
    let area = d3.area()
                .x(d => xScale(d[0]))
                .y0(height - marginBottom)
                .y1(d => yScale(d[1]));
    
    svg.append("path")
        .datum(points)
        .attr("clip-path", "url(" + plotSelector + ")")
        .attr("fill", "#FF5500")
        .attr("stroke", "none")
        .attr("opacity", "0.2")
        .attr("d", area);
    
    svg.append("path")
        .datum(points)
        .attr("clip-path", "url(" + plotSelector + ")")
        .attr("fill", "none")
        .attr("stroke", "#FF5500")
        .attr("stroke-width", 2)
        .attr("d", line);
                            
    viz.append(svg.node());
}

plotBetaPdf("#viz-beliefs");
document.querySelectorAll("#viz-beliefs ~ input")
        .forEach(_ => addEventListener(
            "input", _ => plotBetaPdf("#viz-beliefs")
        ))