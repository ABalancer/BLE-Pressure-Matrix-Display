from bleak import BleakScanner
from bleak import BleakClient
import asyncio
import multiprocessing
import struct


def decode_matrix_data(byte_array):
    # Assuming each element is a 12-bit unsigned integer (2 bytes)
    element_size = 2
    num_rows = 16
    num_cols = 16

    # Unpack the byte array into a flat list of integers
    unpacked_matrix_data = struct.unpack('<' + 'H' * (len(byte_array) // element_size), byte_array)
    # Reshape the flat list into a 2D matrix
    matrix_data = [unpacked_matrix_data[i:i+num_rows] for i in range(0, len(unpacked_matrix_data), num_cols)]

    return matrix_data


def start_asyncio(target, args):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    if isinstance(args, tuple) and len(args) > 0:
        loop.run_until_complete(target(*args))
    else:
        loop.run_until_complete(target(args))
    loop.close()


def process_handler(target, args):
    process = multiprocessing.Process(target=start_asyncio, args=(target, args))
    process.start()
    return process


# Function to update the listbox with detected devices
async def device_scanner(queue, lock, data_availability):
    try:
        devices = await BleakScanner.discover()
        if devices:
            devices_2d = []
            for device in devices:
                if device.name:
                    device_name = device.name
                else:
                    device_name = "None"
                devices_2d.append((device.address, device_name))
            lock.acquire()
            queue.put(devices_2d)
            data_availability.value = 1
            lock.release()
            print("Devices Found")
    except Exception:
        print("Bleak Scanner Failed - Possible Issue with Bluetooth Adapter")
        data_availability.value = 2


async def connect(queue, lock, connected, data_availability, device_address, characteristic):
    client = BleakClient(device_address)
    try:
        await client.connect()
        print("Connected to the device")
        lock.acquire()
        connected.value = 1
        lock.release()
        while connected.value:
            byte_array = await client.read_gatt_char(characteristic)
            matrix_data = decode_matrix_data(byte_array)
            lock.acquire()
            queue.put(matrix_data)
            data_availability.value = 1
            lock.release()

    except Exception as e:
        lock.acquire()
        queue.put(None)
        data_availability.value = 0
        lock.release()
        print(f"Connection terminated on BLE Device\nError: {e}")

    finally:
        if client.is_connected:
            await client.disconnect()
            print("Disconnected from the device")
