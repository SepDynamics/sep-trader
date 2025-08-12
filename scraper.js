// This is a template for a Node.js scraper on morph.io (https://morph.io)

var cheerio = require("cheerio");
var request = require("request");
var sqlite3 = require("sqlite3").verbose();

function initDatabase(callback) {
    // Set up sqlite database.
    var db = new sqlite3.Database("data.sqlite", (err) => {
        if (err) {
            console.error("Error opening database:", err);
            process.exit(1);
        }
    });

    db.serialize(function() {
        // Create tables with useful fields
        db.run(`CREATE TABLE IF NOT EXISTS data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            url TEXT,
            description TEXT,
            author TEXT,
            date_scraped DATETIME,
            last_modified DATETIME
        )`);

        db.run(`CREATE TABLE IF NOT EXISTS metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT UNIQUE,
            value TEXT,
            last_updated DATETIME
        )`);

        callback(db);
    });
}

function updateRow(db, data) {
    return new Promise((resolve, reject) => {
        const stmt = db.prepare(`
            INSERT INTO data (
                title, url, description, author,
                date_scraped, last_modified
            ) VALUES (?, ?, ?, ?, datetime('now'), datetime('now'))
        `);

        stmt.run(
            data.title || null,
            data.url || null,
            data.description || null,
            data.author || null,
            (err) => {
                if (err) {
                    console.error("Error inserting data:", err);
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

function fetchPage(url) {
    return new Promise((resolve, reject) => {
        const options = {
            url: url,
            headers: {
                'User-Agent': 'Mozilla/5.0 (compatible; MorphWebScraper/1.0)'
            },
            timeout: 10000
        };

        request(options, function(error, response, body) {
            if (error) {
                console.error("Error requesting page:", error);
                reject(error);
            } else if (response.statusCode !== 200) {
                reject(new Error(`HTTP ${response.statusCode} for ${url}`));
            } else {
                resolve(body);
            }
        });
    });
}

async function run(db) {
    try {
        // Example: scraping a hypothetical blog or news site
        const body = await fetchPage("https://morph.io");
        const $ = cheerio.load(body);

        // Find and process each article/item
        const promises = $("div.media-body").map(async function() {
            const $element = $(this);
            
            const data = {
                title: $element.find("span.p-name").text().trim(),
                url: $element.find("a").attr("href"),
                description: $element.find("p").text().trim(),
                author: $element.find(".author").text().trim() || "Unknown"
            };

            if (data.title) {
                await updateRow(db, data);
            }
        }).get();

        await Promise.all(promises);
        
        // Read and display results
        const results = await readRows(db);
        console.log("Latest scraped items:", results);

    } catch (error) {
        console.error("Error during scraping:", error);
    } finally {
        // Ensure database is properly closed
        db.close((err) => {
            if (err) {
                console.error("Error closing database:", err);
            }
        });
    }
}

// Initialize database and start scraping
initDatabase(run);