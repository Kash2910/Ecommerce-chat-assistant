// Import Google's Gemini chat model and embeddings for AI text generation and vector creation
import { ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings } from "@langchain/google-genai"
import { StructuredOutputParser } from "@langchain/core/output_parsers"
import { MongoClient } from "mongodb"
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb"
import { z } from "zod"
import "dotenv/config"

// ✅ Debug log at startup
console.log("🚀 Seed script started")

// MongoDB client
const client = new MongoClient(process.env.MONGODB_ATLAS_URI as string)

// Gemini model
const llm = new ChatGoogleGenerativeAI({
  model: "gemini-1.5-flash",
  temperature: 0.7,
  apiKey: process.env.GOOGLE_API_KEY,
})

// Schema
const itemSchema = z.object({
  item_id: z.string(),
  item_name: z.string(),
  item_description: z.string(),
  brand: z.string(),
  manufacturer_address: z.object({
    street: z.string(),
    city: z.string(),
    state: z.string().nullable(),
    postal_code: z.string(),
    country: z.string(),
  }),
  prices: z.object({
    full_price: z.number(),
    sale_price: z.number(),
  }),
  categories: z.array(z.string()),
  user_reviews: z.array(
    z.object({
      review_date: z.string(),
      rating: z.number(),
      comment: z.string(),
    })
  ),
  notes: z.string(),
})

type Item = z.infer<typeof itemSchema>

// Parser for multiple items
const parser = StructuredOutputParser.fromZodSchema(z.array(itemSchema))

// Ensure DB + collection exist
async function setupDatabaseAndCollection(): Promise<void> {
  const db = client.db("inventory_database")
  const collections = await db.listCollections({ name: "items" }).toArray()

  if (collections.length === 0) {
    await db.createCollection("items")
    console.log("✅ Created 'items' collection")
  } else {
    console.log("ℹ️ 'items' collection already exists")
  }
}

// Try to create vector search index (skip if not supported)
async function tryCreateVectorSearchIndex(): Promise<void> {
  try {
    const db = client.db("inventory_database")
    const collection = db.collection("items")

    console.log("⚙️ Attempting to create vector search index...")
    await collection.createSearchIndex({
      name: "vector_index",
      type: "vectorSearch",
      definition: {
        fields: [
          {
            type: "vector",
            path: "embedding",
            numDimensions: 768,
            similarity: "cosine",
          },
        ],
      },
    })
    console.log("✅ Vector search index created")
  } catch (e: any) {
    if (e.codeName === "SearchNotEnabled") {
      console.log("⚠️ Skipping vector search index creation (not supported on free tier)")
    } else {
      console.error("❌ Failed to create vector search index:", e)
    }
  }
}

// Generate synthetic data
async function generateSyntheticData(count: number): Promise<Item[]> {
  const prompt = `Generate ${count} furniture items with realistic details.
  Each must include item_id, item_name, item_description, brand, manufacturer_address, prices, categories, user_reviews, notes.
  ${parser.getFormatInstructions()}`

  console.log(`📝 Generating ${count} items...`)
  const response = await llm.invoke(prompt)
  return parser.parse(response.content as string)
}

// Create summary for embeddings
async function createItemSummary(item: Item): Promise<string> {
  const manufacturerDetails = `Made in ${item.manufacturer_address.country}`
  const categories = item.categories.join(", ")
  const userReviews = item.user_reviews
    .map((review) => `Rated ${review.rating} on ${review.review_date}: ${review.comment}`)
    .join(" ")
  const basicInfo = `${item.item_name} ${item.item_description} from ${item.brand}`
  const price = `Full price: ${item.prices.full_price} USD, Sale: ${item.prices.sale_price} USD`
  const notes = item.notes

  return `${basicInfo}. Manufacturer: ${manufacturerDetails}. Categories: ${categories}. Reviews: ${userReviews}. Price: ${price}. Notes: ${notes}`
}

// Main seeding function
async function seedDatabase(): Promise<void> {
  try {
    await client.connect()
    await client.db("admin").command({ ping: 1 })
    console.log("✅ Connected to MongoDB!")

    await setupDatabaseAndCollection()
    await tryCreateVectorSearchIndex()

    const db = client.db("inventory_database")
    const collection = db.collection("items")

    await collection.deleteMany({})
    console.log("🧹 Cleared existing data from items collection")

    const embeddings = new GoogleGenerativeAIEmbeddings({
      apiKey: process.env.GOOGLE_API_KEY,
      modelName: "text-embedding-004",
    })

    // Generate 10 items at a time
    const TOTAL_ITEMS = 20
    const CHUNK_SIZE = 5

    for (let i = 0; i < TOTAL_ITEMS; i += CHUNK_SIZE) {
      const items = await generateSyntheticData(CHUNK_SIZE)

      for (const record of items) {
        const summary = await createItemSummary(record)
        const doc = { pageContent: summary, metadata: { ...record } }

        await MongoDBAtlasVectorSearch.fromDocuments([doc], embeddings, {
          collection,
          indexName: "vector_index",
          textKey: "embedding_text",
          embeddingKey: "embedding",
        })

        console.log(`✅ Inserted item ${record.item_id}`)
      }
    }

    console.log("🎉 Database seeding completed successfully!")
  } catch (error) {
    console.error("❌ Error seeding database:", error)
  } finally {
    await client.close()
  }
}

// Run
seedDatabase().catch(console.error)
