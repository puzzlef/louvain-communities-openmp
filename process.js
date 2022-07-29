const fs = require('fs');
const os = require('os');

const RGRAPH = /^Loading graph .*\/(.*?)\.mtx \.\.\./m;
const RORDER = /^order: (\d+) size: (\d+) \{\}/m;
const RRESLT = /^\[(.+?) ms; (\d+) iters\.\] \[(.+?) err\.\] (\w+)/m;




// *-FILE
// ------

function readFile(pth) {
  var d = fs.readFileSync(pth, 'utf8');
  return d.replace(/\r?\n/g, '\n');
}

function writeFile(pth, d) {
  d = d.replace(/\r?\n/g, os.EOL);
  fs.writeFileSync(pth, d);
}




// *-CSV
// -----

function writeCsv(pth, rows) {
  var cols = Object.keys(rows[0]);
  var a = cols.join()+'\n';
  for (var r of rows)
    a += [...Object.values(r)].map(v => `"${v}"`).join()+'\n';
  writeFile(pth, a);
}




// *-LOG
// -----

function readLogLine(ln, data, state) {
  if (RGRAPH.test(ln)) {
    var [, graph] = RGRAPH.exec(ln);
    if (!data.has(graph)) data.set(graph, []);
    state = {graph};
  }
  else if (RORDER.test(ln)) {
    var [, order, size] = RORDER.exec(ln);
    state.order = parseFloat(order);
    state.size  = parseFloat(size);
  }
  else if (RRESLT.test(ln)) {
    var [, time, iterations, error, technique] = RRESLT.exec(ln);
    data.get(state.graph).push({
      graph: state.graph,
      order: state.order,
      size:  state.size,
      time:       parseFloat(time),
      iterations: parseFloat(iterations),
      error:      parseFloat(error),
      technique:  technique,
    });
  }
  return state;
}

function readLog(pth) {
  var text  = readFile(pth);
  var lines = text.split('\n');
  var data  = new Map();
  var state = null;
  for (var ln of lines)
    state = readLogLine(ln, data, state);
  return data;
}




// PROCESS-*
// ---------


function processCsv(data) {
  var a = [];
  for (var rows of data.values())
    a.push(...rows);
  return a;
}




// MAIN
// ----

function main(cmd, log, out) {
  var data = readLog(log);
  switch (cmd) {
    case 'csv':
      var rows = processCsv(data);
      writeCsv(out, rows);
      break;
    default:
      console.error(`error: "${cmd}"?`);
      break;
  }
}
main(...process.argv.slice(2));
