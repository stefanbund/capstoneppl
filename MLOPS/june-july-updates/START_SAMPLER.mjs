// const capsWriter = require('./capsWriter.cjs');
import capsWriter from './capsWriter.cjs';

// const d3 = require('d3-array')
// import * as d3 from 'd3-array';
// const fs = require('fs');
import fs from 'fs';
// const axios = require('axios');
import axios from 'axios'

const { existsSync, mkdirSync } = fs;
// const path = require('path');
import path from 'path';
import { fileURLToPath } from 'url';
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const join = path.join;

async function getSupportedCryptocurrencies() {
    try {
        const response = await axios.get('https://api.pro.coinbase.com/currencies');
        const currencies = response.data;
        const tradePairs = currencies
            .filter(currency => currency.details.type === 'crypto')
            .map(currency => `${currency.id}-USD`);
        return tradePairs;
    } catch (error) {
        console.error('Error fetching supported cryptocurrencies:', error.message);
        return [];
    }
}

async function main() {
  const tradePairs = await getSupportedCryptocurrencies();
  tradePairs.forEach(tradePair => {
      const interval = 4; //how often, in seconds, we query for orderbook via coinbse websocket/orderbook
      var caps = {};
      let oba = new orderBookAnalysis(tradePair)
  });
}

main();

// function createFolders(tradePairs) {
//     console.log("length of trade pairs array: " + tradePairs.length)
//     tradePairs.forEach(tradePair => {
//         const folderPath = join(__dirname, tradePair);
//         if (!fs.existsSync(folderPath)) {
//             fs.mkdirSync(folderPath);
//             console.log(`Created folder: ${tradePair}`);
//         } else {
//             // console.log(`Folder already exists: ${tradePair}`);
//             // dirCount++
//         }
//     });
// }



//----------------------------------------------oba
// const capsWriter = require('./capsWriter.cjs')
// Import the createRequire function from the 'module' module
// import { createRequire } from 'module';

// Create a 'require' function specific to this module
// const require = createRequire(import.meta.url);

// Now you can use 'require' just like in CommonJS
let minBidder = [];//captyures min bid value
let minBid = 0.0; //the minimum value of bid offers in the relevant Bids array
var bidAskCaps = {}; //return for chaser and integration with gladiator version 1.6; they need a cap number for bids and asks
var exits = []; //compiled after each use of histogram, contains mp, timestamp epoch, entry, exit, used to gauge predictability
let obj = {};//used to store time, as a global across the app
import pkg from 'coinbase-pro';
const { PublicClient } = pkg;
const publicClient = new PublicClient();
const pair = '' // 'AVAX-USD';
var currentAsks = []; //all bid values, without separation
var currentBids = []; //all ask  values, without separation
const pctRange = .0036; //prior setting, jan/dec .0037; //string representation of the values within 25% of mid point we collect/analyze herein
var diff; //the metric we will pass to the graph
var midPointPriceForRecords = 0.0; //what we'll save as the current price, or mid point of order book
var currencyPairs = []; //init via initCurrencyPairs()
var totalBidCap = 0.0;
var totalAskCap = 0.0;
var totalBidVolume = 0.0;
var totalAskVolume = 0.0;
import { mode, sum } from "stats-lite"; //stats lite npm, https://www.npmjs.com/package/stats-lite
var preBidWall, preSellWall = 0.0; //value of orders leading up to bid or sell wall area , calc'd in analyzeRangedArray
var askMode, bidMode = 0.0; //sell wall and buy wall, respectively (most occuring value within each array)
var askSBV, bidSBV = 0.0; // sum biv value, sum ask value under the bid or sell walls (capitalization whale must buy to trigger the wall)........DEP 11-26
var bidWall = 0; //on 201, to help zero it out between uses / runs
var sumUnderBidWall = 0.0; //gets reused in the 200's
var getInterval; //do interval
var currentPair = ""; //currency pair we are currently analyzing, coinbase USD set
import { exit } from 'process';
var probableDirection = ""; //which way the market will turn, bull or bear, depending on the compared values, bidMode(down) vs askMode (up)
const csvBids = []; //what to access to get relevant bids and asks, for histogram
const csvAsks = [];

class orderBookAnalysis{
  constructor(pair){
    this.pair = pair
    console.log("init with trade pair " + this.pair)
    this.initOBA(this.pair)
  }

 getCaps () //used to prove the output of oba, then used in production to write caps to csv
{
  var oa = [];
  oa.push(bidAskCaps);
}
initOBA(tradePair) // pair would be 'AVAX-USD' sent by caller
{
  let delay2 = 3000; //
  let timer2 = setInterval(function request()
  {
    publicClient.getProductOrderBook(
      tradePair, { level: 3 },
      (error, response, book) =>
      {
        
        if (book)
        {
          var bids = [];
          book.bids.forEach(function (item, index, array)
          {
            bids.push(item);
          });
          currentBids = bids;
          var asks = [];
          book.asks.forEach(function (i, index, array)
          {
            asks.push(i);
          });
          currentAsks = asks;
          sumBidVolume(bids); //GOOD
          sumAskVolume(asks); //GOOD
          totalBidsinUSD(bids);
          totalAsksinUSD(asks);
          getPctUpDown(bids, asks, pctRange); //for experiment 1 --

        }
      })
    }, delay2);



  function totalBidsinUSD(bids)
  {
    var bidCap = 0.0; //will mutiply each item by amt * bid price
    bids.forEach(function (item)
    {
      var thisCap = parseFloat(item[0]) * parseFloat(item[1]);
      bidCap = bidCap + thisCap;
    });
    totalBidCap = bidCap.toFixed(2);
  }

   function totalAsksinUSD(asks)
  {
    var askCap = 0.0;
    asks.forEach(function (item)
    {
      var thisC = parseFloat(item[0]) * parseFloat(item[1]);
      askCap = askCap + thisC;
    });
    totalAskCap = askCap.toFixed(2);
  }

   function sumBidVolume(bids)
   { //SUM OF ALL ORDERS TO ...
    var sumBids = 0.0; //for starters
    bids.forEach(function (item)
    { 
      sumBids = parseFloat(item[1]) + parseFloat(sumBids);
    });
    totalBidVolume = sumBids.toFixed(2);
  }

  function sumAskVolume(asks)
  { 
    var sumAsks = 0.0; //for starters
    asks.forEach(function (item)
    { 
      sumAsks = parseFloat(item[1]) + parseFloat(sumAsks);
    });
    totalAskVolume = sumAsks.toFixed(2);
  }

  function  getPctUpDown(bids, asks, pct)
  {

    if (bids.length > 0 && asks.length >0){
      var tfbid = bids[0][0] - (bids[0][0] * pct);
      var tfask = (asks[0][0] * pct) + parseFloat(asks[0][0]); //should get us our 25% range value
      searchOrderBookWithinRanges(tfbid, tfask, currentAsks, currentBids);
    }
    else{
      console.log("for trade pair: " + tradePair)

      console.log(bids)
    }
      
    
  }

  //1. create a provisional array to house all bids within the 25% limit, assign to global bid filtered array
  //console.log("length of ask and bid arrays: " + currentAsks.length + ", " + currentBids.length); //ok
  //2. creater a provisional array to house all bids within the 25% limit, assign to global ask filtered array

  //locates bids and ask orders within the range we seek, delivers relevantBids and relevantAsks, within range
  function searchOrderBookWithinRanges(tfbid, tfask, currentAsks, currentBids)
  {
    var relevantAsks = [];
    var relevantBids = [];
    for (var i = 0; i < currentAsks.length; i++)
    {
      midPointPriceForRecords = parseFloat(currentAsks[0][0]); //first price at top of stack
      if (currentAsks[i][0] <= tfask)
      {
        relevantAsks.push(currentAsks[i]); //lines up all asks within 25% of mid point price
      }
    }

    for (var i = 0; i < currentBids.length; i++)
    {
      if (parseFloat(currentBids[i][0]) >= tfbid && parseFloat(currentBids[i][0]) > 0.0)
      {
        relevantBids.push(currentBids[i]);
      }
    }

    triggerHisto(); //done as alternative to writeRelevantArraysToCsv
  }

  function triggerHisto()
  {

    let capsArr = []; //push to csv
    let co = {}; //caps object
    co.symbol = tradePair;
    co.bc = totalBidCap;
    co.ac = totalAskCap;
    co.tbv = totalBidVolume;
    co.tav = totalAskVolume;
    co.time = Date.now();
    co.mp = midPointPriceForRecords;
    capsArr.push(co);//most recent matched element. 
    capsWriter.writeMatches(capsArr);
}//end initOBA()
}//end class?
}
