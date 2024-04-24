import { pdf as betaPdf } from 'https://cdn.jsdelivr.net/gh/stdlib-js/stats-base-dists-beta@esm/index.mjs';

import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7/+esm";



function plotBetaPdf(plotSelector){
    const viz = document.querySelector(plotSelector);
    const sliderAlpha = document.querySelector(plotSelector + "~ input.alpha");
    const sliderBeta = document.querySelector(plotSelector + "~ input.beta");

    const width = viz.offsetWidth;
    const height = width / 1.8;
    const marginTop = 20;
    const marginRight = 20;
    const marginBottom = 30;
    const marginLeft = 40;
    const xPrecision = 0.01;


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
                    .domain([0, 20])
                    .range([height - marginBottom, marginTop]);
    
    svg.append("g")
        .attr("transform", `translate(0,${height - marginBottom})`)
        .call(d3.axisBottom(xScale));
        
    svg.append("g")
        .attr("transform", `translate(${marginLeft},0)`)
        .call(d3.axisLeft(yScale));

    let alpha = sliderAlpha.valueAsNumber;
    let beta = sliderBeta.valueAsNumber;
    const points = [];
    for (let x = 0; x <= 1; x += xPrecision){
        let y = betaPdf(x, alpha, beta);
        points.push([x, y]);
    }
    
    console.log(points);

    let line = d3.line()
                .x(d => xScale(d[0]))
                .y(d => yScale(d[1]));
    
    svg.append("path")
        .datum(points)
        .attr("clip-path", "url(" + plotSelector + ")")
        .attr("fill", "none")
        .attr("stroke", "teal")
        .attr("stroke-width", 2)
        .attr("d", line);
                            
    viz.append(svg.node());
}

plotBetaPdf("#viz-beliefs");
document.querySelectorAll("#viz-beliefs ~ input")
        .forEach(_ => addEventListener(
            "input", _ => plotBetaPdf("#viz-beliefs")
        ))