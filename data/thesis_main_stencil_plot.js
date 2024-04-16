const fs = require('fs');
const plt = require('nodeplotlib');
const calculateCorrelation = require('calculate-correlation');

const data = JSON.parse(fs.readFileSync("/home/lksri/repos/trigsolve-data-munch/thesis_postprocesssed_results_stencils_v2.json"));

const data_100 = data["/stencil7_100_input.mtx.input"];
const data_10000 = data["/stencil7_10000_input.mtx.input"];
const data_1000000 = data["/stencil7_1000000_input.mtx.input"];

function generateBarChart100(sample) {
    const r = sample;
    const csr_opt = Math.min(r.thinned, r.thinned_spin, r.syncfree, r.syncfree_spin, r.syncfree_warpperrow, r.udiag, r.thinned_udiag, r.udiag_leg);
    const csc_opt = Math.min(r.csc, r.cscb);
    const cusparse = r.sparselib;

    let rem = [{a: r.rdpi, b: "rdpi"}, {a: r.rspi, b: "rspi"}, {a: r.winv, b: "winv"}, {a: r.wvar, b: "wvar"}, {a: r.rscpi, b: "rscpi"}, {a: r.ppi, b: "ppi"}];
    rem.sort((x, y) => x.a - y.a)

    let p = { x: ["csr_opt", "csc_opt", "cusparse", rem[0].b], y: [1/csr_opt, 1/csc_opt, 1/cusparse, 1/rem[0].a], name: "5pt stencil, size 100", type: "bar", 
    marker: {color: ["cornflowerblue", "darkgoldenrod", "green", "crimson"]},
    text: [1/csr_opt, 1/csc_opt, 1/cusparse, 1/rem[0].a].map(Math.round).map(String),
    textposition: 'auto',};
    return p;
}

function generateBarChart10000(sample) {
    const r = sample;
    const csr_opt = Math.min(r.thinned, r.thinned_spin, r.syncfree, r.syncfree_spin, r.syncfree_warpperrow, r.udiag, r.thinned_udiag, r.udiag_leg);
    const csc_opt = Math.min(r.csc, r.cscb);
    const cusparse = r.sparselib;

    let rem = [{a: r.rdpi, b: "rdpi"}, {a: r.rspi, b: "rspi"}, {a: r.winv, b: "winv"}, {a: r.wvar, b: "wvar"}, {a: r.rscpi, b: "rscpi"}, {a: r.ppi, b: "ppi"}];
    rem.sort((x, y) => x.a - y.a)

    let p = { x: ["csr_opt", "csc_opt", "cusparse", rem[0].b], y: [1/csr_opt, 1/csc_opt, 1/cusparse, 1/rem[0].a], name: "5pt stencil, size 10000", type: "bar",
    marker: {color: ["cornflowerblue", "darkgoldenrod", "green", "crimson"]},
    text: [1/csr_opt, 1/csc_opt, 1/cusparse, 1/rem[0].a].map(Math.round).map(String),
    textposition: 'auto',};
    return p;
}

function generateBarChart1000000(sample) {
    const r = sample;
    const csr_opt = Math.min(r.thinned, r.thinned_spin, r.syncfree, r.syncfree_spin, r.syncfree_warpperrow, r.udiag, r.thinned_udiag, r.udiag_leg);
    const csc_opt = Math.min(r.csc, r.cscb);
    const cusparse = r.sparselib;

    let rem = [{a: r.rdpi, b: "rdpi"}, {a: r.rspi, b: "rspi"}, {a: r.winv, b: "winv"}, {a: r.wvar, b: "wvar"}, {a: r.rscpi, b: "rscpi"}, {a: r.ppi, b: "ppi"}];
    rem.sort((x, y) => x.a - y.a)

    let p = { x: ["csr_opt", "csc_opt", "cusparse", rem[0].b], y: [1/csr_opt, 1/csc_opt, 1/cusparse, 1/rem[0].a], name: "5pt stencil, size 1000000", type: "bar",
    marker: {color: ["cornflowerblue", "darkgoldenrod", "green", "crimson"]},
    text: [1/csr_opt, 1/csc_opt, 1/cusparse, 1/rem[0].a].map(Math.round).map(String),
    textposition: 'auto',};
    return p;
}

const trace1 = generateBarChart100(data_100);
const trace2 = generateBarChart10000(data_10000);
const trace3 = generateBarChart1000000(data_1000000);

let layout = {
    bargap : 0
}

const pltd = [ 
    trace2
];
// console.log(trace1);

plt.plot(pltd, layout)