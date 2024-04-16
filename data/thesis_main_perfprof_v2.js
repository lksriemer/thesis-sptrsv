const fs = require('fs');
const plt = require('nodeplotlib');


function cleanData(d) {
    let cleaned = [];
    Object.values(d).forEach(e => {
            // let p = Math.log2(parseFloat(e.n / e.nlevels * (1 - e.subdiag)));
            let winv = e.winv;
            let thinned = e.thinned;
            let thinned_spin = e["thinned_spin"];
            let syncfree = parseFloat(e.syncfree);
            let syncfree_spin = parseFloat(e.syncfree_spin);
            let cusparse = parseFloat(e["sparselib"]);
            let ppi = parseFloat(e["ppi"]);
            let rdpi = parseFloat(e["rdpi"]);
            let rspi = parseFloat(e["rspi"]);
            let wvar = parseFloat(e["wvar"]);
            let udiag = parseFloat(e["udiag"]);
            let thinned_udiag = parseFloat(e["thinned_udiag"]);
            let udiag_leg = parseFloat(e["udiag_leg"]);
            let thinned_udiag_leg = parseFloat(e["thinned_udiag_leg"]);
            let syncfree_pt = parseFloat(e["syncfree_pt"]); // persistent threads
            let syncfree_nc = parseFloat(e["syncfree_nc"]); // non-caching
            let csc = parseFloat(e["csc"]);
            let cscb = parseFloat(e["cscb"]);
            let rscpi = parseFloat(e["rscpi"]);
            let rdpig = parseFloat(e["rdpig"]); // fully ginkgo-spmv reliant
            let rdmpi4 = parseFloat(e["rdmpi4"]);
            let rdmpi128 = parseFloat(e["rdmpi128"]);

            let thinned_udiag_opt = Math.min(thinned_udiag, thinned_udiag_leg);
            let udiag_opt = Math.min(udiag, udiag_leg);

            let syncfree_opt = Math.min(udiag_opt, syncfree, syncfree_spin, syncfree_pt);
            let thinned_opt = Math.min(thinned_udiag_opt, thinned, thinned_spin);
            
            let csropt = Math.min(thinned, thinned_spin, thinned_udiag, syncfree, syncfree_spin, udiag, thinned_udiag_leg, udiag_leg);
            
            // '/HTC_336_4438.mtx.input'

            let n  = e.n;

            const arr = [winv, thinned, syncfree, cusparse, rspi, thinned_spin, syncfree_spin, rdpi, 
                wvar, ppi, udiag, thinned_udiag, thinned_udiag_leg, udiag_leg, syncfree_pt, syncfree_nc, csc, cscb, rscpi, rdpig];
            arr.sort(function(a, b) {
              return a - b;
            });

            let time_min = arr[0];
            let time_second_min = arr[1];

            // if (time_min == syncfree && 1.1 * time_min < time_second_min){
            //     console.log("syncfree very best with: ", e.name);
            // }

            // if (time_min == wvar && 1.3 * time_min < time_second_min){
            //     console.log("wvar very best with: ", e.name);
            // }

            // if (time_min == winv && 3 * time_min < time_second_min){
            //     console.log("winv very best with: ", e.name);
            // }

            // if (time_min == thinned && 1.3 * time_min < time_second_min){
            //     console.log("thinned very best with: ", e.name);
            // }

            // if (time_min == rspi && 1.4 * time_min < cusparse){
            //     console.log("rspi very best with: ", e.name);
            // }

            // if (time_min == rdpi && 1.4 * time_min < cusparse){
            //     console.log("rdpi very best with: ", e.name);
            // }

            // if(thinned_udiag_opt >= 1.45 * thinned){
            //     console.log("Thinned udiag really bad at " + e.name);
            // }

            // if(udiag_opt >= 1.5 * syncfree){
            //     console.log("udiag really bad at " + e.name);
            // }

            //  if(syncfree >= 3 * syncfree_spin){
            //     console.log("syncfree non spin really bad at " + e.name);
            // }

            if(1.8 * cscb <= time_second_min){
                console.log("cscb really good at ", e.name);
            }

            // let s = e.fastest_lower_trs / Math.min(e.csrswi_lower_trs, e.csrm8_lower_trs);
            cleaned.push({ n: n, winv: winv, thinned: thinned, syncfree: syncfree, cusparse: cusparse, 
                time_min: time_min, thinned_spin: thinned_spin, syncfree_spin: syncfree_spin, ppi: ppi, 
                rdpi: rdpi, rspi: rspi, wvar: wvar, udiag: udiag, thinned_udiag: thinned_udiag, csropt: csropt,
            thinned_udiag_leg: thinned_udiag_leg, udiag_leg: udiag_leg, thinned_udiag_opt: thinned_udiag_opt, 
            udiag_opt: udiag_opt, syncfree_pt: syncfree_pt, syncfree_opt: syncfree_opt, thinned_opt: thinned_opt, syncfree_nc: syncfree_nc,
            csc: csc, cscb: cscb, rscpi: rscpi, rdpig: rdpig, rdmpi4: rdmpi4, rdmpi128: rdmpi128});
        
    });
    return cleaned;
}


function generatePlotseries1(results) {
    let p = { x: [], y: [], name: "winv"};
    let epsilon = 1e-9;
    for(let i = 1; i < 8; i += 0.01){
        let count = 0;
        results.forEach((r) => {
            if(r.winv / r.time_min < i + epsilon){
                count += 1;
            }
        });
        p.x.push(i);
        p.y.push(count);
    }
    return p;
}

function generatePlotseries2(results) {
    let p = { x: [], y: [], name: "thinned"};
    let epsilon = 1e-9;
    for(let i = 1; i < 8; i += 0.01){
        let count = 0;
        results.forEach((r) => {
            if(r.thinned / r.time_min < i + epsilon){
                count += 1;
            }
        });
        p.x.push(i);
        p.y.push(count);
    }
    return p;
}

function generatePlotseries3(results) {
    let p = { x: [], y: [], name: "thinned_spin"};
    let epsilon = 1e-9;
    for(let i = 1; i < 8; i += 0.01){
        let count = 0;
        results.forEach((r) => {
            if(r.thinned_spin / r.time_min < i + epsilon){
                // if(i == 1){
                //     console.log("thinned_spin best with: ", r.thinned_spin);
                // }
                count += 1;
            }
        });
        p.x.push(i);
        p.y.push(count);
    }
    return p;
}

function generatePlotseries4(results) {
    let p = { x: [], y: [], name: "syncfree"};
    let epsilon = 1e-9;
    for(let i = 1; i < 8; i += 0.01){
        let count = 0;
        results.forEach((r) => {
            if(r.syncfree / r.time_min < i + epsilon){
                count += 1;
            }
        });
        p.x.push(i);
        p.y.push(count);
    }
    return p;
}

function generatePlotseries5(results) {
    let p = { x: [], y: [], name: "syncfree_spin"};
    let epsilon = 1e-9;
    for(let i = 1; i < 8; i += 0.01){
        let count = 0;
        results.forEach((r) => {
            if(r.syncfree_spin / r.time_min < i + epsilon){
                count += 1;
            }
        });
        p.x.push(i);
        p.y.push(count);
    }
    return p;
}

function generatePlotseries6(results) {
    let p = { x: [], y: [], name: "not_cusparse"};
    let epsilon = 1e-9;
    for(let i = 1; i < 8; i += 0.01){
        let count = 0;
        results.forEach((r) => {
            if(Math.min(r.winv, r.thinned, r.syncfree, r.rspi, r.thinned_spin, r.syncfree_spin, r.rdpi, r.
                wvar, r.ppi, r.udiag, r.thinned_udiag, r.thinned_udiag_leg, r.udiag_leg, r.syncfree_pt, 
                r.syncfree_nc, r.csc, r.cscb, r.rscpi, r.rdpig) / r.time_min < i + epsilon){
                count += 1;
            }
        });
        p.x.push(i);
        p.y.push(count);
    }
    return p;
}

function generatePlotseries7(results) {
    let p = { x: [], y: [], name: "cusparse"};
    let epsilon = 1e-9;
    for(let i = 1; i < 8; i += 0.01){
        let count = 0;
        results.forEach((r) => {
            if(r.cusparse / r.time_min < i + epsilon){
                count += 1;
            }
        });
        p.x.push(i);
        p.y.push(count);
    }
    return p;
}

function generatePlotseries8(results) {
    let p = { x: [], y: [], name: "ppi"};
    let epsilon = 1e-9;
    for(let i = 1; i < 8; i += 0.01){
        let count = 0;
        results.forEach((r) => {
            if(r.ppi / r.time_min < i + epsilon){
                count += 1;
            }
        });
        p.x.push(i);
        p.y.push(count);
    }
    return p;
}

function generatePlotseries9(results) {
    let p = { x: [], y: [], name: "rdpi"};
    let epsilon = 1e-9;
    for(let i = 1; i < 8; i += 0.01){
        let count = 0;
        results.forEach((r) => {
            if(r.rdpi / r.time_min < i + epsilon){
                count += 1;
            }
        });
        p.x.push(i);
        p.y.push(count);
    }
    return p;
}

function generatePlotseries10(results) {
    let p = { x: [], y: [], name: "rspi"};
    let epsilon = 1e-9;
    for(let i = 1; i < 8; i += 0.01){
        let count = 0;
        results.forEach((r) => {
            if(r.rspi / r.time_min < i + epsilon){
                count += 1;
            }
        });
        p.x.push(i);
        p.y.push(count);
    }
    return p;
}

function generatePlotseries11(results) {
    let p = { x: [], y: [], name: "wvar"};
    let epsilon = 1e-9;
    for(let i = 1; i < 8; i += 0.01){
        let count = 0;
        results.forEach((r) => {
            if(r.wvar / r.time_min < i + epsilon){
                count += 1;
            }
        });
        p.x.push(i);
        p.y.push(count);
    }
    return p;
}

function generatePlotseries12(results) {
    let p = { x: [], y: [], name: "csr_opt"};
    let epsilon = 1e-9;
    for(let i = 1; i < 8; i += 0.01){
        let count = 0;
        results.forEach((r) => {
            if(r.csropt / r.time_min < i + epsilon){
                count += 1;
            }
        });
        p.x.push(i);
        p.y.push(count);
    }
    return p;
}

function generatePlotseries13(results) {
    let p = { x: [], y: [], name: "syncfree_udiag"};
    let epsilon = 1e-9;
    for(let i = 1; i < 8; i += 0.01){
        let count = 0;
        results.forEach((r) => {
            if(r.udiag / r.time_min < i + epsilon){
                count += 1;
            }
        });
        p.x.push(i);
        p.y.push(count);
    }
    return p;
}

function generatePlotseries14(results) {
    let p = { x: [], y: [], name: "thinned_udiag"};
    let epsilon = 1e-9;
    for(let i = 1; i < 8; i += 0.01){
        let count = 0;
        results.forEach((r) => {
            if(r.thinned_udiag / r.time_min < i + epsilon){
                count += 1;
            }
        });
        p.x.push(i);
        p.y.push(count);
    }
    return p;
}

function generatePlotseries15(results) {
    let p = { x: [], y: [], name: "udiag_opt"};
    let epsilon = 1e-9;
    for(let i = 1; i < 8; i += 0.01){
        let count = 0;
        results.forEach((r) => {
            if(r.udiag_opt / r.time_min < i + epsilon){
                count += 1;
            }
        });
        p.x.push(i);
        p.y.push(count);
    }
    return p;
}

function generatePlotseries16(results) {
    let p = { x: [], y: [], name: "thinned_udiag_opt"};
    let epsilon = 1e-9;
    for(let i = 1; i < 8; i += 0.01){
        let count = 0;
        results.forEach((r) => {
            if(r.thinned_udiag_opt / r.time_min < i + epsilon){
                count += 1;
            }
        });
        p.x.push(i);
        p.y.push(count);
    }
    return p;
}

function generatePlotseries17(results) {
    let p = { x: [], y: [], name: "thinned_opt"};
    let epsilon = 1e-9;
    for(let i = 1; i < 8; i += 0.01){
        let count = 0;
        results.forEach((r) => {
            if(Math.min(r.thinned_udiag_opt, r.thinned, r.thinned_spin) / r.time_min < i + epsilon){
                count += 1;
            }
        });
        p.x.push(i);
        p.y.push(count);
    }
    return p;
}

function generatePlotseries18(results) {
    let p = { x: [], y: [], name: "syncfree_opt"};
    let epsilon = 1e-9;
    for(let i = 1; i < 8; i += 0.01){
        let count = 0;
        results.forEach((r) => {
            if(Math.min(r.udiag_opt, r.syncfree, r.syncfree_spin) / r.time_min < i + epsilon){
                count += 1;
            }
        });
        p.x.push(i);
        p.y.push(count);
    }
    return p;
}

function generatePlotseries19(results) {
    let p = { x: [], y: [], name: "csc"};
    let epsilon = 1e-9;
    for(let i = 1; i < 8; i += 0.01){
        let count = 0;
        results.forEach((r) => {
            if(r.csc / r.time_min < i + epsilon){
                count += 1;
            }
        });
        p.x.push(i);
        p.y.push(count);
    }
    return p;
}

function generatePlotseries20(results) {
    let p = { x: [], y: [], name: "cscb"};
    let epsilon = 1e-9;
    for(let i = 1; i < 8; i += 0.01){
        let count = 0;
        results.forEach((r) => {
            if(r.cscb / r.time_min < i + epsilon){
                count += 1;
            }
        });
        p.x.push(i);
        p.y.push(count);
    }
    return p;
}

function generatePlotseries21(results) {
    let p = { x: [], y: [], name: "rscpi"};
    let epsilon = 1e-9;
    for(let i = 1; i < 8; i += 0.01){
        let count = 0;
        results.forEach((r) => {
            if(r.rscpi / r.time_min < i + epsilon){
                count += 1;
            }
        });
        p.x.push(i);
        p.y.push(count);
    }
    return p;
}

function generatePlotseries22(results) {
    let p = { x: [], y: [], name: "rdpig"};
    let epsilon = 1e-9;
    for(let i = 1; i < 8; i += 0.01){
        let count = 0;
        results.forEach((r) => {
            if(r.rdpig / r.time_min < i + epsilon){
                count += 1;
            }
        });
        p.x.push(i);
        p.y.push(count);
    }
    return p;
}

function generatePlotseries23(results) {
    let p = { x: [], y: [], name: "rdmpi4"};
    let epsilon = 1e-9;
    for(let i = 1; i < 8; i += 0.01){
        let count = 0;
        results.forEach((r) => {
            if(r.rdmpi4 / r.time_min < i + epsilon){
                count += 1;
            }
        });
        p.x.push(i);
        p.y.push(count);
    }
    return p;
}

function generatePlotseries24(results) {
    let p = { x: [], y: [], name: "rdmpi128"};
    let epsilon = 1e-9;
    for(let i = 1; i < 8; i += 0.01){
        let count = 0;
        results.forEach((r) => {
            if(r.rdmpi128 / r.time_min < i + epsilon){
                count += 1;
            }
        });
        p.x.push(i);
        p.y.push(count);
    }
    return p;
}

function generateRelativePlotseries(results, nameA, nameB) {
    let p = { x: [], y: [], name: "profile " + nameA + " vs " + nameB};
    let epsilon = 1e-9;
    const cutoff = 8;
    for(let i = 1; i < cutoff; i += 0.01){
        let count = 0;
        results.forEach((r) => {
            // if(r.n >= 200000){
                if(r[nameA] / r[nameB] < i + epsilon){
                count += 1;
            }
            // }
            
        });
        p.x.push(i);
        p.y.push(count);
    }
    return p;
}

// const data = JSON.parse(fs.readFileSync('/home/lksri/repos/trigsolve-data-munch/full_aggregated_wiv8wv1.json',
//     { encoding: 'utf8', flag: 'r' }));
const data = JSON.parse(fs.readFileSync('/home/lksri/repos/trigsolve-data-munch/thesis_postprocesssed_results_v9.json', { encoding: 'utf8', flag: 'r' }));

// let series = generateCorrelationSeries(data);
// let cor = calculateCorrelation(series[0], series[1]);
// console.log(cor);

// dataInspection(data);

const cleaned = cleanData(data);
// console.log(cleaned);

const trace1 = generatePlotseries1(cleaned); // winv
const trace2 = generatePlotseries2(cleaned); // thinned
const trace3 = generatePlotseries3(cleaned); // thinned_spin
const trace4 = generatePlotseries4(cleaned); // syncfree
const trace5 = generatePlotseries5(cleaned); // syncfree_spin
const trace6 = generatePlotseries6(cleaned); // not_cusparse
const trace7 = generatePlotseries7(cleaned); // cusparse
const trace8 = generatePlotseries8(cleaned); // ppi
const trace9 = generatePlotseries9(cleaned); // rdpi
const trace10 = generatePlotseries10(cleaned); // rspi
const trace11 = generatePlotseries11(cleaned); // wvar
const trace12 = generatePlotseries12(cleaned); // csr_opt
const trace13 = generatePlotseries13(cleaned); // syncfree_udiag
const trace14 = generatePlotseries14(cleaned); // thinned_udiag
const trace15 = generatePlotseries15(cleaned); // udiag_opt
const trace16 = generatePlotseries16(cleaned); // thinned_udiag_opt
const trace17 = generatePlotseries17(cleaned); // thinned_opt
const trace18 = generatePlotseries18(cleaned); // syncfree_opt
const trace19 = generatePlotseries19(cleaned); // csc
const trace20 = generatePlotseries20(cleaned); // cscb
const trace21 = generatePlotseries21(cleaned); // rscpi
const trace22 = generatePlotseries22(cleaned); // rdpig
const trace23 = generatePlotseries23(cleaned); // rdmpi4
const trace24 = generatePlotseries24(cleaned); // rdmpi128

const relTrace1 = generateRelativePlotseries(cleaned, "winv", "thinned");
const relTrace2 = generateRelativePlotseries(cleaned, "thinned", "winv");
const relTrace3 = generateRelativePlotseries(cleaned, "cusparse", "thinned");
const relTrace4 = generateRelativePlotseries(cleaned, "thinned", "cusparse");
const relTrace5 = generateRelativePlotseries(cleaned, "thinned", "syncfree");
const relTrace6 = generateRelativePlotseries(cleaned, "syncfree", "thinned");

const relTrace7 = generateRelativePlotseries(cleaned, "thinned", "thinned_udiag_opt");
const relTrace8 = generateRelativePlotseries(cleaned, "thinned_udiag_opt", "thinned");
const relTrace9 = generateRelativePlotseries(cleaned, "syncfree", "udiag_opt");
const relTrace10 = generateRelativePlotseries(cleaned, "udiag_opt", "syncfree");

const relTrace11 = generateRelativePlotseries(cleaned, "csropt", "winv");
const relTrace12 = generateRelativePlotseries(cleaned, "winv", "csropt");

const relTrace13 = generateRelativePlotseries(cleaned, "csropt", "wvar");
const relTrace14 = generateRelativePlotseries(cleaned, "wvar", "csropt");

const relTrace15 = generateRelativePlotseries(cleaned, "csropt", "rspi");
const relTrace16 = generateRelativePlotseries(cleaned, "rspi", "csropt");
const relTrace17 = generateRelativePlotseries(cleaned, "cusparse", "rspi");
const relTrace18 = generateRelativePlotseries(cleaned, "rspi", "cusparse");

const relTrace19 = generateRelativePlotseries(cleaned, "cusparse", "csropt");
const relTrace20 = generateRelativePlotseries(cleaned, "csropt", "cusparse");

const relTrace21 = generateRelativePlotseries(cleaned, "csropt", "rdpi");
const relTrace22 = generateRelativePlotseries(cleaned, "rdpi", "csropt");
const relTrace23 = generateRelativePlotseries(cleaned, "cusparse", "rdpi");
const relTrace24 = generateRelativePlotseries(cleaned, "rdpi", "cusparse");

const relTrace25 = generateRelativePlotseries(cleaned, "udiag", "udiag_leg");
const relTrace26 = generateRelativePlotseries(cleaned, "udiag_leg", "udiag");

const relTrace27 = generateRelativePlotseries(cleaned, "thinned_udiag", "thinned_udiag_leg");
const relTrace28 = generateRelativePlotseries(cleaned, "thinned_udiag_leg", "thinned_udiag");

const relTrace29 = generateRelativePlotseries(cleaned, "syncfree", "syncfree_pt");
const relTrace30 = generateRelativePlotseries(cleaned, "syncfree_pt", "syncfree");

const relTrace31 = generateRelativePlotseries(cleaned, "syncfree_opt", "thinned_opt");
const relTrace32 = generateRelativePlotseries(cleaned, "thinned_opt", "syncfree_opt");

const relTrace33 = generateRelativePlotseries(cleaned, "syncfree", "syncfree_spin");
const relTrace34 = generateRelativePlotseries(cleaned, "syncfree_spin", "syncfree");

const relTrace35 = generateRelativePlotseries(cleaned, "thinned", "thinned_spin");
const relTrace36 = generateRelativePlotseries(cleaned, "thinned_spin", "thinned");

const relTrace37 = generateRelativePlotseries(cleaned, "syncfree_opt", "syncfree_nc");
const relTrace38 = generateRelativePlotseries(cleaned, "syncfree_nc", "syncfree_opt");

const relTrace39 = generateRelativePlotseries(cleaned, "csropt", "csc");
const relTrace40 = generateRelativePlotseries(cleaned, "csc", "csropt");

const relTrace41 = generateRelativePlotseries(cleaned, "csropt", "cscb");
const relTrace42 = generateRelativePlotseries(cleaned, "cscb", "csropt");

const relTrace43 = generateRelativePlotseries(cleaned, "csropt", "rdpig");
const relTrace44 = generateRelativePlotseries(cleaned, "rdpig", "csropt");

const relTrace45 = generateRelativePlotseries(cleaned, "cusparse", "rdpig");
const relTrace46 = generateRelativePlotseries(cleaned, "rdpig", "cusparse");

const relTrace47 = generateRelativePlotseries(cleaned, "rdpig", "rdpi");
const relTrace48 = generateRelativePlotseries(cleaned, "rdpi", "rdpig");


// const pltd = [ 
//     trace1, 
//     // trace2, 
//     // trace3, 
//     // trace4, 
//     // trace5, 
//     trace6, 
//     trace7,
//     // relTrace29, relTrace30
//     // trace8, 
//     trace9, 
//     trace10, 
//     trace11, 
//     trace12, 
//     // trace13, trace14
//     // trace15, trace16
//     // trace17, trace18
//     trace19, trace20, 
//     // trace21, 
//     trace22
// ];
const pltd = [ 
    // trace1, 
    // trace2, 
    // trace3, 
    // trace4, 
    // trace5, 
    // trace6, 
    // trace7,
    relTrace47, relTrace48
    // trace8, 
    // trace9, 
    // trace10, 
    // trace11, 
    // trace12, 
    // trace13, trace14
    // trace15, trace16
    // trace17, trace18
    // trace19, trace20, 
    // trace21, 
    // trace22
];

plt.plot(pltd)