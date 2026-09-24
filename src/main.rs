#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    uni_api_native::run().await
}
