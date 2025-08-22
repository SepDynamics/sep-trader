// This is a template for a Node.js scraper on morph.io (https://morph.io)

var fs = require('fs').promises;
var path = require('path');
var sqlite3 = require("sqlite3").verbose();
var { execFile } = require('child_process');
var { promisify } = require('util');
const execFileAsync = promisify(execFile);

async function getLastCommitInfo(filePath) {
    try {
        const { stdout } = await execFileAsync('git', [
            'log',
            '-1',
            '--format=%H|%s|%aI',
            '--',
            filePath
        ], { cwd: process.cwd() });

        const [hash, message, date] = stdout.trim().split('|');
        return { hash, message, date };
    } catch (error) {
        console.warn(`Could not get commit info for ${filePath}:`, error.message);
        return {
            hash: '',
            message: '',
            date: new Date().toISOString()
        };
    }
}

function initDatabase(callback) {
    // Set up sqlite database.
    var db = new sqlite3.Database("data.sqlite", (err) => {
        if (err) {
            console.error("Error opening database:", err);
            process.exit(1);
        }
    });

    db.serialize(function() {
        // Create tables for repository data
        db.run(`CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE,
            type TEXT,
            size INTEGER,
            last_commit_hash TEXT,
            last_commit_message TEXT,
            last_commit_date DATETIME,
            date_scraped DATETIME DEFAULT CURRENT_TIMESTAMP
        )`);

        db.run(`CREATE TABLE IF NOT EXISTS commits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hash TEXT UNIQUE,
            author TEXT,
            message TEXT,
            date DATETIME,
            files_changed INTEGER,
            additions INTEGER,
            deletions INTEGER,
            date_scraped DATETIME DEFAULT CURRENT_TIMESTAMP
        )`);

        callback(db);
    });
}

function updateFileRow(db, data) {
    return new Promise((resolve, reject) => {
        const stmt = db.prepare(`
            INSERT OR REPLACE INTO files (
                path, type, size, last_commit_hash,
                last_commit_message, last_commit_date
            ) VALUES (?, ?, ?, ?, ?, ?)
        `);

        stmt.run(
            data.path,
            data.type,
            data.size || 0,
            data.lastCommit.hash,
            data.lastCommit.message,
            data.lastCommit.date,
            (err) => {
                if (err) {
                    console.error("Error inserting file data:", err);
                    reject(err);
                } else {
                    resolve();
                }
            }
        );
        stmt.finalize();
    });
}

function updateCommitRow(db, data) {
    return new Promise((resolve, reject) => {
        const stmt = db.prepare(`
            INSERT OR REPLACE INTO commits (
                hash, author, message, date,
                files_changed, additions, deletions
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        `);

        stmt.run(
            data.hash,
            data.author,
            data.message,
            data.date,
            data.filesChanged,
            data.additions,
            data.deletions,
            (err) => {
                if (err) {
                    console.error("Error inserting commit data:", err);
                    reject(err);
                } else {
                    resolve();
                }
            }
        );
        stmt.finalize();
    });
}

function readRows(db) {
    return new Promise((resolve, reject) => {
        const results = [];
        db.each(
            `SELECT * FROM data ORDER BY date_scraped DESC LIMIT 10`,
            (err, row) => {
                if (err) {
                    console.error("Error reading row:", err);
                    reject(err);
                } else {
                    results.push(row);
                }
            },
            (err, count) => {
                if (err) {
                    reject(err);
                } else {
                    console.log(`Found ${count} rows`);
                    resolve(results);
                }
            }
        );
    });
}

// Rate limiting helper
const rateLimiter = {
    queue: [],
    lastRequest: 0,
    minDelay: 1000, // Minimum 1 second between requests
    
    async wait() {
        const now = Date.now();
        const timeToWait = Math.max(0, this.lastRequest + this.minDelay - now);
        if (timeToWait > 0) {
            await new Promise(resolve => setTimeout(resolve, timeToWait));
        }
        this.lastRequest = Date.now();
    }
};

async function fetchApi(path, retries = 3) {
    await rateLimiter.wait();
    
    return new Promise((resolve, reject) => {
        const options = {
            url: `https://api.github.com/repos/sepdynamics/sep-trader${path}`,
            headers: {
                'User-Agent': 'SEPTraderScraper/1.0',
                'Accept': 'application/vnd.github.v3+json'
            },
            timeout: 30000
        };

        request(options, async function(error, response, body) {
            if (error) {
                console.error("Error requesting API:", error);
                if (retries > 0) {
                    console.log(`Retrying... (${retries} attempts left)`);
                    await new Promise(resolve => setTimeout(resolve, 2000));
                    return resolve(await fetchApi(path, retries - 1));
                }
                reject(error);
            } else if (response.statusCode === 429) {
                const resetTime = response.headers['x-ratelimit-reset'];
                const waitTime = resetTime ? (resetTime * 1000 - Date.now()) : 60000;
                console.log(`Rate limited. Waiting ${Math.ceil(waitTime/1000)} seconds...`);
                await new Promise(resolve => setTimeout(resolve, waitTime));
                return resolve(await fetchApi(path, retries));
            } else if (response.statusCode !== 200) {
                reject(new Error(`HTTP ${response.statusCode} for ${options.url}`));
            } else {
                resolve(JSON.parse(body));
            }
        });
    });
}

async function processDirectory(db, dirPath = '.') {
    try {
        const entries = await fs.readdir(dirPath, { withFileTypes: true });
        const contents = [];

        for (const entry of entries) {
            // Skip hidden files and certain directories
            if (entry.name.startsWith('.') ||
                ['node_modules', 'build', 'dist'].includes(entry.name)) {
                continue;
            }

            const fullPath = path.join(dirPath, entry.name);
            const relativePath = path.relative('.', fullPath);

            const fileData = {
                path: relativePath,
                type: entry.isDirectory() ? 'dir' : 'file',
                size: 0,
                lastCommit: { hash: '', message: '', date: '' }
            };

            if (entry.isFile()) {
                const stats = await fs.stat(fullPath);
                fileData.size = stats.size;
                fileData.lastCommit = await getLastCommitInfo(relativePath);
            }

            await updateFileRow(db, fileData);
            contents.push(fileData);

            if (entry.isDirectory()) {
                await processDirectory(db, fullPath);
            }
        }

        return contents;
    } catch (error) {
        console.error(`Error processing directory ${dirPath}:`, error);
        return [];
    }
}

async function run(db) {
    let processedFiles = 0;
    let processedCommits = 0;

    try {
        console.log("Starting scrape of sep-trader repository");
        console.log("Phase 1: Scanning repository structure...");
        
        // Get repository contents
        const files = await processDirectory(db);
        processedFiles = files.length;
        console.log(`Processed ${processedFiles} top-level items`);
        
        console.log("\nPhase 2: Fetching recent commits...");
        // Get only the 10 most recent commits to stay within rate limits
        const commits = await fetchApi('/commits?per_page=10');
        
        for (const commit of commits) {
            try {
                const commitData = {
                    hash: commit.sha,
                    message: commit.commit.message,
                    author: commit.commit.author.name,
                    date: commit.commit.author.date,
                    filesChanged: 0,
                    additions: 0,
                    deletions: 0
                };

                // Skip commit details for merge commits to reduce API calls
                if (!commit.commit.message.startsWith('Merge')) {
                    const details = await fetchApi(`/commits/${commit.sha}`);
                    commitData.filesChanged = details.files?.length || 0;
                    commitData.additions = details.stats?.additions || 0;
                    commitData.deletions = details.stats?.deletions || 0;
                }

                await updateCommitRow(db, commitData);
                processedCommits++;
                console.log(`Processed commit ${processedCommits}/${commits.length}: ${commit.sha.substring(0, 7)}`);
            } catch (error) {
                if (error.message.includes('429')) {
                    console.log('Rate limit reached, saving progress and exiting...');
                    break;
                }
                console.warn(`Skipping commit ${commit.sha}: ${error.message}`);
            }
        }
        
        console.log("\nScraping Summary:");
        console.log(`- Files processed: ${processedFiles}`);
        console.log(`- Commits processed: ${processedCommits}`);
        console.log("Scraping completed successfully");

    } catch (error) {
        console.error("Error during scraping:", error);
        console.log("Partial results have been saved to the database");
    } finally {
        db.close((err) => {
            if (err) {
                console.error("Error closing database:", err);
            }
        });
    }
}

// Initialize database and start scraping
initDatabase(run);