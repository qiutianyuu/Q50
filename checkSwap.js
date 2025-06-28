const { Web3 } = require('web3');
const web3 = new Web3('https://bsc-dataseed.binance.org/');

const pairAddress = '0x2312b9068adeaea57845d62a3bf79dbd6af21085';

// 扩展ABI以包含RemoveLiquidity事件
const pairAbi = [
    {
        "constant": true,
        "inputs": [],
        "name": "getReserves",
        "outputs": [
            { "name": "_reserve0", "type": "uint112" },
            { "name": "_reserve1", "type": "uint112" },
            { "name": "_blockTimestampLast", "type": "uint32" }
        ],
        "type": "function"
    },
    {
        "anonymous": false,
        "inputs": [
            { "indexed": true, "name": "sender", "type": "address" },
            { "indexed": false, "name": "amount0In", "type": "uint256" },
            { "indexed": false, "name": "amount1In", "type": "uint256" },
            { "indexed": false, "name": "amount0Out", "type": "uint256" },
            { "indexed": false, "name": "amount1Out", "type": "uint256" },
            { "indexed": true, "name": "to", "type": "address" }
        ],
        "name": "Swap",
        "type": "event"
    }
];

const pairContract = new web3.eth.Contract(pairAbi, pairAddress);

// 查池子数据
async function getPoolReserves() {
    try {
        const reserves = await pairContract.methods.getReserves().call();
        // 先转为字符串，再转为数字
        const reserve0 = web3.utils.fromWei(reserves._reserve0.toString(), 'ether');
        const reserve1 = web3.utils.fromWei(reserves._reserve1.toString(), 'ether');
        console.log(`WBNB: ${reserve0}, Token: ${reserve1}`);
    } catch (error) {
        console.error("Error fetching reserves:", error);
    }
}

// 查过去一小时的Swap
async function getPastSwaps() {
    try {
        const latestBlock = await web3.eth.getBlockNumber();
        const fromBlock = latestBlock - 1200; // 大约1小时（BSC每块3秒）
        const events = await pairContract.getPastEvents('Swap', {
            fromBlock,
            toBlock: 'latest'
        });

        for (let event of events) {
            try {
                const block = await web3.eth.getBlock(event.blockNumber);
                const timestamp = Number(block.timestamp);
                // 使用 web3.utils.fromWei 来处理大数
                const amount0In = web3.utils.fromWei(event.returnValues.amount0In.toString(), 'ether');
                const amount1Out = web3.utils.fromWei(event.returnValues.amount1Out.toString(), 'ether');
                console.log(
                    `Swap: WBNB In: ${amount0In}, Token Out: ${amount1Out}, Time: ${new Date(timestamp * 1000).toLocaleString()}`
                );
            } catch (error) {
                console.error("Error processing event:", error);
                continue;
            }
        }
    } catch (error) {
        console.error("Error fetching swaps:", error);
    }
}

// 主函数跑起来
(async () => {
    await getPoolReserves();
    await getPastSwaps();})();
