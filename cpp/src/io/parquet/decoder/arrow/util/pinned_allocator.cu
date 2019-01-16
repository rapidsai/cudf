#include "pinned_allocator.cuh"

	cudaError_t CachingPinnedAllocator::pinnedAllocate(
			void            **d_ptr,            ///< [out] Reference to pointer to the allocation
			size_t          bytes     )
	{
		*d_ptr                          = NULL;

		cudaError_t error               = cudaSuccess;


		// Create a block descriptor for the requested allocation
		bool found = false;
		BlockDescriptor search_key;

		NearestPowerOf(search_key.bin, search_key.bytes, bin_growth, bytes);

		if (search_key.bin > max_bin)
		{
			// Bin is greater than our maximum bin: allocate the request
			// exactly and give out-of-bounds bin.  It will not be cached
			// for reuse when returned.
			search_key.bin      = INVALID_BIN;
			search_key.bytes    = bytes;
		}
		else
		{
			// Search for a suitable cached allocation: lock
			mutex.lock();

			if (search_key.bin < min_bin)
			{
				// Bin is less than minimum bin: round up
				search_key.bin      = min_bin;
				search_key.bytes    = min_bin_bytes;
			}

			// Iterate through the range of cached blocks on the same device in the same bin
			CachedBlocks::iterator block_itr = cached_blocks.lower_bound(search_key);
			while ((block_itr != cached_blocks.end())
					&& (block_itr->bin == search_key.bin))
			{
				// To prevent races with reusing blocks returned by the host but still
				// in use by the device, only consider cached blocks that are
				// either (from the active stream) or (from an idle stream)

				// Reuse existing cache block.  Insert into live blocks.
				found = true;
				search_key = *block_itr;

				live_blocks.insert(search_key);

				// Remove from free blocks
				cached_bytes.free -= search_key.bytes;
				cached_bytes.live += search_key.bytes;

				cached_blocks.erase(block_itr);


				block_itr++;
			}

			// Done searching: unlock
			mutex.unlock();
		}

		// Allocate the block if necessary
		if (!found)
		{

			// Attempt to allocate
			if (cudaMallocHost((void **)&search_key.d_ptr, search_key.bytes) != cudaSuccess)
			{

				error = cudaSuccess;    // Reset the error we will return
				cudaGetLastError();     // Reset CUDART's error

				// Lock
				mutex.lock();

				// Iterate the range of free blocks on the same device
				BlockDescriptor free_key;
				CachedBlocks::iterator block_itr = cached_blocks.lower_bound(free_key);

				while ((block_itr != cached_blocks.end()))
				{
					// No need to worry about synchronization with the device: cudaFreeHost is
					// blocking and will synchronize across all kernels executing
					// on the current device

					// Free device memory and destroy stream event.
					error = cudaFreeHost(block_itr->d_ptr);
					if(error != cudaSuccess){
					//	std::cout<<"could not free from host";
						break;
					}

					// Reduce balance and erase entry
					cached_bytes.free -= block_itr->bytes;


					cached_blocks.erase(block_itr);

					block_itr++;
				}

				// Unlock
				mutex.unlock();

				// Return under error
				if (error) return error;

				// Try to allocate again
				error = cudaMallocHost((void **)&search_key.d_ptr, search_key.bytes);
				if(error != cudaSuccess){
					return error;
				}

			}

			// Insert into live blocks
			mutex.lock();
			live_blocks.insert(search_key);
			cached_bytes.live += search_key.bytes;
			mutex.unlock();


		}

		// Copy device pointer to output parameter
		*d_ptr = search_key.d_ptr;

		return error;
	}


	cudaError_t CachingPinnedAllocator::pinnedFree(
				void*           d_ptr)
		{
			cudaError_t error               = cudaSuccess;



			// Lock
			mutex.lock();

			// Find corresponding block descriptor
			bool recached = false;
			BlockDescriptor search_key(d_ptr);
			BusyBlocks::iterator block_itr = live_blocks.find(search_key);
			if (block_itr != live_blocks.end())
			{
				// Remove from live blocks
				search_key = *block_itr;
				live_blocks.erase(block_itr);
				cached_bytes.live -= search_key.bytes;

				// Keep the returned allocation if bin is valid and we won't exceed the max cached threshold
				if ((search_key.bin != INVALID_BIN) && (cached_bytes.free + search_key.bytes <= max_cached_bytes))
				{
					// Insert returned allocation into free blocks
					recached = true;
					cached_blocks.insert(search_key);
					cached_bytes.free += search_key.bytes;

				}
			}

			// Unlock
			mutex.unlock();


			if (recached)
			{
				// Insert the ready event in the associated stream (must have current device set properly)
				//TODO: see if we have to do anything here to handle concurrency
			}
			else
			{
				// Free the allocation from the runtime and cleanup the event.
				error = cudaFreeHost(d_ptr);
				if (error != cudaSuccess) return error;

			}

			return error;
		}


	cudaError_t CachingPinnedAllocator::FreeAllCached()
		{
			cudaError_t error         = cudaSuccess;

			mutex.lock();

			while (!cached_blocks.empty())
			{
				// Get first block
				CachedBlocks::iterator begin = cached_blocks.begin();



				// Free device memory
				error = cudaFreeHost(begin->d_ptr);
				if (error != cudaSuccess) break;

				// Reduce balance and erase entry
				cached_bytes.free -= begin->bytes;

				cached_blocks.erase(begin);
			}

			mutex.unlock();


			return error;
		}
