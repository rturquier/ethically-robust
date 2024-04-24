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

/**
 * Beta probability density function impementation
 * using logarithms, no factorials involved.
 * Overcomes the problem with large integers.
 * Code taken from https://github.com/royhzq/betajs.
 */
function betaPdf(x, a, b) {
    // Beta probability density function impementation
    // using logarithms, no factorials involved.
    // Overcomes the problem with large integers
    return Math.exp(lnBetaPDF(x, a, b))
}
function lnBetaPDF(x, a, b) {
        // Log of the Beta Probability Density Function
    return ((a-1)*Math.log(x) + (b-1)*Math.log(1-x)) - lnBetaFunc(a,b)
}
function lnBetaFunc(a, b) {
		// Log Beta Function
	  // ln(Beta(x,y))
    let foo = 0.0;

    for (let i=0; i<a-2; i++) {
        foo += Math.log(a-1-i);
    }
    for (let i=0; i<b-2; i++) {
        foo += Math.log(b-1-i);
    }
    for (let i=0; i<a+b-2; i++) {
        foo -= Math.log(a+b-1-i);
    }
    return foo
}


function plotBetaPdf(plotSelector){
    const viz = document.querySelector(plotSelector);
    const sliderMean = document.querySelector(plotSelector + "~ input.mean");
    const sliderSigma = document.querySelector(plotSelector + "~ input.sigma");

    const width = viz.offsetWidth;
    const height = width / 1.8;
    const marginTop = 20;
    const marginRight = 20;
    const marginBottom = 30;
    const marginLeft = 40;
    const xPrecision = 0.001;


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
                    .domain([0, 10])
                    .range([height - marginBottom, marginTop]);
    
    svg.append("g")
        .attr("transform", `translate(0,${height - marginBottom})`)
        .call(d3.axisBottom(xScale));
        
    svg.append("g")
        .attr("transform", `translate(${marginLeft},0)`)
        .call(d3.axisLeft(yScale).tickFormat(""));

    const mean = sliderMean.valueAsNumber;
    const sigma = sliderSigma.valueAsNumber;
    const variance = sigma**2;
    const [alpha, beta] = momentsToParameters(mean, variance);

    const points = [];
    for (let x = 0 + xPrecision; x <= 1 - xPrecision; x += xPrecision){
        let y = betaPdf(x, alpha, beta);
        points.push([x, y]);
    }
    
    console.log(mean, variance, alpha, beta);

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